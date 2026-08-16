/*
 * cnlunar.h — 中国农历/黄历 (Chinese lunar almanac) single-header C library.
 *
 * A C99 port of the Python `cnlunar` package
 * (https://github.com/opn48/cnlunar), which is based on
 * 《钦定协纪辨方书》(Qinding Xiejibianfangshu).
 *
 * Capabilities: lunar date conversion (1901–2099; same data range as the Python
 * original), 八字 (BaZi), 神煞宜忌 (auspicious/inauspicious gods and 宜/忌 lists),
 * 24 solar terms, zodiacs, 二十八星宿, 九宫飞星, 胎神, 时辰 luck, holidays.
 *
 * All data is static; no allocation, no I/O, thread-safe, reentrant.
 *
 * Parity with the Python original (verified over every 10th day, 1901–2100,
 * all four option combos): every one of the 62 output fields matches — all
 * calendar / 八字 / zodiac / solar-term / 星宿 / 时辰 / 方位 / 九宫 / 胎神 /
 * 彭祖 / 节日 fields are byte-identical, and the 神煞 god-name and 宜/忌 item
 * lists match as sets. (Item order within those lists is compared set-wise
 * because the Python original's final list order is hash-seeded.)
 *
 * ---------------------------------------------------------------
 * HOW TO USE
 * ---------------------------------------------------------------
 * Define CNLUNAR_IMPLEMENTATION in exactly ONE translation unit before
 * including this header, then call cnlunar_calculate():
 *
 *     #define CNLUNAR_IMPLEMENTATION
 *     #include "cnlunar.h"
 *
 *     int main(void) {
 *         cnlunar_result r;
 *         int rc = cnlunar_calculate(&r, 2026, 3, 9, 12, 30, CNLUNAR_DEFAULT);
 *         if (rc != 0) return rc;
 *         printf("%s\n", r.year8char);   // year8char == "丙午"
 *         return 0;
 *     }
 *
 * License: MIT (see LICENSE). Copyright (c) 2026 cnlunar C port.
 */

#ifndef CNLUNAR_H_INCLUDED
#define CNLUNAR_H_INCLUDED

#include <stddef.h>  /* size_t（公共 API cnlunar_calculate_ws 使用） */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Options                                                             */
/* ------------------------------------------------------------------ */
/* Option bit flags passed to cnlunar_calculate(). Defaults = 0.      */
#define CNLUNAR_GODTYPE_CNLUNAR    0x0001u  /* else '8char' algorithm   */
#define CNLUNAR_YEAR8CHAR_LICHUN   0x0002u  /* else 'year' algorithm     */
#define CNLUNAR_YEARGOD_NODUTY     0x0004u  /* else 'duty' (岁* gods)    */

#define CNLUNAR_DEFAULT 0u

/* ------------------------------------------------------------------ */
/* Return codes                                                        */
/* ------------------------------------------------------------------ */
#define CNLUNAR_OK                0
#define CNLUNAR_ERR_DATE         (-1)  /* invalid date components       */
#define CNLUNAR_ERR_RANGE        (-2)  /* outside 1901..2100 data table */
#define CNLUNAR_ERR_NULL         (-3)  /* NULL result pointer           */
#define CNLUNAR_ERR_WORKSPACE    (-4)  /* workspace 过小                */

/* ------------------------------------------------------------------ */
/* Capacity bounds (output lists can never exceed these)               */
/* ------------------------------------------------------------------ */
#define CNLUNAR_LIST_MAX   256         /* max entries in any list       */
#define CNLUNAR_ITEM_MAX   64          /* max bytes per UTF-8 item      */
#define CNLUNAR_STR_MAX    128         /* max bytes per free string     */

/* A counted list of UTF-8 strings. */
typedef struct {
    int n;
    char it[CNLUNAR_LIST_MAX][CNLUNAR_ITEM_MAX];
} cnlunar_list;

/* cnlunar_calculate_ws() 所需的 workspace 最小字节数：
 * 计算过程中 3 个瞬态清单槽（cnl_angel_demon 的 allgods/dibt/rm*，互相排斥或
 * 短生命周期）会从 workspace 中切取，另留 1 槽余量。嵌入式设备请用
 * platform_malloc(PSRAM) 提供该工作区；0 栈需求（任务栈可小至 12KB）。
 */
#define CNLUNAR_WORKSPACE_MIN (4 * sizeof(cnlunar_list))

/* A solar (month, day) pair. */
typedef struct {
    int month, day;
} cnlunar_md;

/* Full almanac result — one cnlunar_calculate() fills this in. */
typedef struct {
    /* -- inputs (echoed) -- */
    int year, month, day, hour, minute;
    unsigned options;

    /* -- lunar date (农历) -- */
    int lunar_year;               /* lunar year number                  */
    int lunar_month;              /* 1..12                              */
    int lunar_day;                /* 1..30                              */
    int is_leap_month;            /* 1 if lunar_month is the leap month */
    int lunar_month_long;         /* 1 = 大月 (30 day), 0 = 小月         */
    int month_days[3];            /* [month, leap_month, leap_day] days */
    int span_days;                /* days from 春节                      */
    char lunar_year_cn[16];        /* 一九零零                           */
    char lunar_month_cn[16];      /* 冬月小 / 闰五月大                  */
    char lunar_day_cn[8];         /* 十一                              */
    char phase_of_moon[8];        /* 朔/望/上弦/下弦 or empty           */

    /* -- 八字 -- */
    char year8char[8];            /* 年柱, e.g. 丙午                    */
    char month8char[8];           /* 月柱                              */
    char day8char[8];             /* 日柱                              */
    char twohour8char[8];         /* 时柱                              */
    cnlunar_list twohour8char_list;   /* 13 个时辰干支                  */
    int day_heavenly_earth_num;   /* 日柱在六十甲子中的序号              */
    int twohour_num;              /* 时辰序号 0..12                      */
    int day_heaven_num, day_earth_num;
    int month_heaven_num, month_earth_num;
    int year_heaven_num, year_earth_num;

    /* -- 季节/月型 -- */
    int lunar_season_type;        /* 0=仲 1=季 2=孟                    */
    int lunar_season_num;         /* 0=春 1=夏 2=秋 3=冬               */
    char lunar_month_type[4];     /* 仲/季/孟                          */
    char lunar_season[4];         /* 春/夏/秋/冬                       */
    char lunar_season_name[8];    /* 仲夏 etc                          */

    /* -- 星座/星次/星期 -- */
    char star_zodiac[16];         /* 摩羯座 .. 射手座                  */
    char today_east_zodiac[8];    /* 星次                              */
    char week_day_cn[12];          /* 星期一 .. 星期日                  */

    /* -- 二十四节气 -- */
    char today_solar_terms[8];    /* 今日节气, "无" if none            */
    int next_solar_num;           /* 0..23 下一节气序号                 */
    char next_solar_term[8];      /* 下一节气名称                       */
    cnlunar_md next_solar_term_date;
    int next_solar_term_year;     /* 节气所属年（可能 +1）               */
    struct {                      /* 今年全部节气, 24 项               */
        char name[8];
        cnlunar_md date;
    } this_year_solar_terms[24];

    /* -- 生肖冲煞 -- */
    char chinese_year_zodiac[4];  /* 生肖                             */
    char chinese_zodiac_clash[16];/* 虎日冲猴                          */
    char zodiac_mark6[4];         /* 六合                               */
    cnlunar_list zodiac_mark3;    /* 三合 2 项                          */
    char zodiac_win[4], zodiac_lose[4];

    /* -- 建除十二神/值神 -- */
    char today12day_officer[4];   /* 建除, e.g. 除                     */
    char today12day_god[8];       /* 青龙..勾陈                        */
    char today12day_name[12];      /* 黄道日 / 黑道日                   */
    char today28star[12];         /* 廿八宿, e.g. 角木蛟               */

    /* -- 节日 -- */
    char holidays_legal[64];      /* 法定假日, comma-joined            */
    char holidays_other[80];      /* 其他阳历节日                      */
    char holidays_lunar[48];      /* 其他农历节日                      */

    /* -- 彭祖百忌 -- */
    char peng_taboo[64];

    /* -- 五行/纳音/九宫/吉神方位/胎神 -- */
    cnlunar_list today5elements;  /* 8-15 项                           */
    char the9flystar[16];         /* 9 位飞星数                        */
    cnlunar_list lucky_gods_direction;  /* 喜神/财神/福神/阳贵/阴贵    */
    char fetal_god[24];           /* 胎神                              */

    /* -- 时辰 -- */
    cnlunar_list twohour_lucky;   /* 13 项 吉/凶                      */
    char meridians[8];            /* 子午流注当令器官，如 心           */
    char meridian_note[CNLUNAR_STR_MAX]; /* 子午流注说明             */
    char meridian_yi[64];         /* 子午流注 宜                      */
    char meridian_ji[64];         /* 子午流注 忌                      */

    /* -- 神煞宜忌 -- */
    cnlunar_list good_god_name;   /* 吉神                              */
    cnlunar_list bad_god_name;    /* 凶煞                              */
    cnlunar_list good_thing;      /* 宜                                */
    cnlunar_list bad_thing;       /* 忌                                */
    int today_level;              /* -1..5 等第                         */
    char today_level_name[CNLUNAR_STR_MAX];
    int thing_level;              /* 宜忌等第 0从宜不从忌..3诸事皆忌    */
    char thing_level_name[16];    /* 从宜不从忌.. 诸事皆忌             */
    int is_de;                    /* 遇德                              */
    char day_overall[8];          /* 当日吉凶：大吉/吉/平/凶            */
} cnlunar_result;

/* Compute the full almanac for local wall-clock (year, month, day, hour,
 * minute) with the given option bits. Returns 0 on success or one of the
 * CNLUNAR_ERR_* codes. On error, *out is left untouched.
 *
 * Supported: years 1901–2100 (solar dates that map into the lunar data
 * table), matching the Python original, which likewise cannot convert dates
 * after lunar year 2099 (its table has 199 entries because the last entry
 * is missing). Out-of-range returns CNLUNAR_ERR_RANGE.
 */
int cnlunar_calculate(cnlunar_result *out, int year, int month, int day,
                      int hour, int minute, unsigned options);

/* 嵌入式版本：与 cnlunar_calculate 完全一致的结果，但不占用大栈（其内部不再
 * 使用 145KB 栈上临时结构），而是将计算过程中的瞬态清单（约3×16KB）切取自调用方
 * 提供的 workspace（须 >= CNLUNAR_WORKSPACE_MIN 字节）。结果直接写入 *out。
 * 同样返回 0 或 CNLUNAR_ERR_*。计算失败时 *out 被清零/部分填充，请勿使用。
 */
int cnlunar_calculate_ws(cnlunar_result *out, int year, int month, int day,
                         int hour, int minute, unsigned options,
                         void *workspace, size_t workspace_size);

#ifdef __cplusplus
}
#endif

#endif /* CNLUNAR_H_INCLUDED */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */

/* ================================================================== */
/*                            IMPLEMENTATION                          */
/* ================================================================== */
#ifdef CNLUNAR_IMPLEMENTATION
/* ========= GENERATED data tables (do not edit by hand) ========= */
/* NOTE: 指针数组一律写 `const char *const`（指针本身也限 const）。
 * ESP32 工具链会把【指针未限 const 的】静态指针表（含重定位）放入内部 DRAM
 * (.data/.data.rel.ro)，而 1D const 字符表落 .rodata(flash)。全表约 5.9KB，
 * 曾挤占 DMA 堆导致启动时 "Failed to alloc frame buffers!"。新增指针表时务必
 * 保持指针级 const。用户新增数据表如不遵守会重新引入该问题。 */

static const char *const cnl_SOLAR_TERMS_NAME[] = {
    "小寒",
    "大寒",
    "立春",
    "雨水",
    "惊蛰",
    "春分",
    "清明",
    "谷雨",
    "立夏",
    "小满",
    "芒种",
    "夏至",
    "小暑",
    "大暑",
    "立秋",
    "处暑",
    "白露",
    "秋分",
    "寒露",
    "霜降",
    "立冬",
    "小雪",
    "大雪",
    "冬至",
};

static const char *const cnl_EAST_ZODIAC[] = {
    "玄枵",
    "娵訾",
    "降娄",
    "大梁",
    "实沈",
    "鹑首",
    "鹑火",
    "鹑尾",
    "寿星",
    "大火",
    "析木",
    "星纪",
};

static const char *const cnl_G60[] = {
    "甲子",
    "乙丑",
    "丙寅",
    "丁卯",
    "戊辰",
    "己巳",
    "庚午",
    "辛未",
    "壬申",
    "癸酉",
    "甲戌",
    "乙亥",
    "丙子",
    "丁丑",
    "戊寅",
    "己卯",
    "庚辰",
    "辛巳",
    "壬午",
    "癸未",
    "甲申",
    "乙酉",
    "丙戌",
    "丁亥",
    "戊子",
    "己丑",
    "庚寅",
    "辛卯",
    "壬辰",
    "癸巳",
    "甲午",
    "乙未",
    "丙申",
    "丁酉",
    "戊戌",
    "己亥",
    "庚子",
    "辛丑",
    "壬寅",
    "癸卯",
    "甲辰",
    "乙巳",
    "丙午",
    "丁未",
    "戊申",
    "己酉",
    "庚戌",
    "辛亥",
    "壬子",
    "癸丑",
    "甲寅",
    "乙卯",
    "丙辰",
    "丁巳",
    "戊午",
    "己未",
    "庚申",
    "辛酉",
    "壬戌",
    "癸亥",
};

static const char *const cnl_NAYIN30[] = {
    "海中金",
    "炉中火",
    "大林木",
    "路旁土",
    "剑锋金",
    "山头火",
    "涧下水",
    "城头土",
    "白蜡金",
    "杨柳木",
    "井泉水",
    "屋上土",
    "霹雳火",
    "松柏木",
    "长流水",
    "砂中金",
    "山下火",
    "平地木",
    "壁上土",
    "金箔金",
    "覆灯火",
    "天河水",
    "大驿土",
    "钗钏金",
    "桑柘木",
    "大溪水",
    "砂中土",
    "天上火",
    "石榴木",
    "大海水",
};

static const char *const cnl_STAR28[] = {
    "角木蛟",
    "亢金龙",
    "氐土貉",
    "房日兔",
    "心月狐",
    "尾火虎",
    "箕水豹",
    "斗木獬",
    "牛金牛",
    "女土蝠",
    "虚日鼠",
    "危月燕",
    "室火猪",
    "壁水貐",
    "奎木狼",
    "娄金狗",
    "胃土雉",
    "昴日鸡",
    "毕月乌",
    "觜火猴",
    "参水猿",
    "井木犴",
    "鬼金羊",
    "柳土獐",
    "星日马",
    "张月鹿",
    "翼火蛇",
    "轸水蚓",
};

static const char *const cnl_PENG_TABOO[] = {
    "甲不开仓 财物耗散",
    "乙不栽植 千株不长",
    "丙不修灶 必见灾殃",
    "丁不剃头 头必生疮",
    "戊不受田 田主不祥",
    "己不破券 二比并亡",
    "庚不经络 织机虚张",
    "辛不合酱 主人不尝",
    "壬不泱水 更难提防",
    "癸不词讼 理弱敌强",
    "子不问卜 自惹祸殃",
    "丑不冠带 主不还乡",
    "寅不祭祀 神鬼不尝",
    "卯不穿井 水泉不香",
    "辰不哭泣 必主重丧",
    "巳不远行 财物伏藏",
    "午不苫盖 屋主更张",
    "未不服药 毒气入肠",
    "申不安床 鬼祟入房",
    "酉不会客 醉坐颠狂",
    "戌不吃犬 作怪上床",
    "亥不嫁娶 不利新郎",
};

static const char *const cnl_CHINESE_ZODIAC[] = {
    "鼠",
    "牛",
    "虎",
    "兔",
    "龙",
    "蛇",
    "马",
    "羊",
    "猴",
    "鸡",
    "狗",
    "猪",
};

static const char *const cnl_12_DAYGOD[] = {
    "青龙",
    "明堂",
    "天刑",
    "朱雀",
    "金贵",
    "天德",
    "白虎",
    "玉堂",
    "天牢",
    "玄武",
    "司命",
    "勾陈",
};

static const char *const cnl_STAR_ZODIAC[] = {
    "摩羯座",
    "水瓶座",
    "双鱼座",
    "白羊座",
    "金牛座",
    "双子座",
    "巨蟹座",
    "狮子座",
    "处女座",
    "天秤座",
    "天蝎座",
    "射手座",
};

static const char *const cnl_12_OFFICER_CHARS = "建除满平定执破危成收开闭";

static const int cnl_STAR_ZODIAC_DATE[12][2] = {
    {1, 20},
    {2, 19},
    {3, 21},
    {4, 21},
    {5, 21},
    {6, 22},
    {7, 23},
    {8, 23},
    {9, 23},
    {10, 23},
    {11, 23},
    {12, 23},
};

/* 方位 */
static const char *const cnl_DIRECTION[8] = {
    "正北",
    "东北",
    "正东",
    "东南",
    "正南",
    "西南",
    "正西",
    "西北",
};
static const char *const cnl_TRIGRAM_8 = "坎艮震巽离坤兑乾";
static const char *const cnl_lucky_dir = "艮乾坤离巽艮乾坤离巽";
static const char *const cnl_wealth_dir = "艮艮坤坤坎坎震震离离";
static const char *const cnl_mascot_dir = "坎坤乾巽艮坎坤乾巽艮";
static const char *const cnl_sunnoble_dir = "坤坤兑乾艮坎离艮震巽";
static const char *const cnl_moonnoble_dir = "艮坎乾兑坤坤艮离巽震";

static const char *const cnl_WEEKDAY[] = {
    "星期一",
    "星期二",
    "星期三",
    "星期四",
    "星期五",
    "星期六",
    "星期日",
};

static const char *const cnl_MONTH_NAME[] = {
    "正月",
    "二月",
    "三月",
    "四月",
    "五月",
    "六月",
    "七月",
    "八月",
    "九月",
    "十月",
    "冬月",
    "腊月",
};

static const char *const cnl_DAY_NAME[] = {
    "初一",
    "初二",
    "初三",
    "初四",
    "初五",
    "初六",
    "初七",
    "初八",
    "初九",
    "初十",
    "十一",
    "十二",
    "十三",
    "十四",
    "十五",
    "十六",
    "十七",
    "十八",
    "十九",
    "二十",
    "廿一",
    "廿二",
    "廿三",
    "廿四",
    "廿五",
    "廿六",
    "廿七",
    "廿八",
    "廿九",
    "三十",
};

static const char *const cnl_UPPER_NUM[] = {
    "零",
    "一",
    "二",
    "三",
    "四",
    "五",
    "六",
    "七",
    "八",
    "九",
};

static const int cnl_ENC_VECTOR[24] = {
    4,
    19,
    3,
    18,
    4,
    19,
    4,
    19,
    4,
    20,
    4,
    20,
    6,
    22,
    6,
    22,
    6,
    22,
    7,
    22,
    6,
    21,
    6,
    21,
};

static const unsigned long long cnl_SOLAR[200] = {
    0x6aaaa6aa9a5aULL,
    0xaaaaaabaaa6aULL,
    0xaaabbabbafaaULL,
    0x5aa665a65aabULL,
    0x6aaaa6aa9a5aULL,
    0xaaaaaaaaaa6aULL,
    0xaaabbabbafaaULL,
    0x5aa665a65aabULL,
    0x6aaaa6aa9a5aULL,
    0xaaaaaaaaaa6aULL,
    0xaaabbabbafaaULL,
    0x5aa665a65aabULL,
    0x6aaaa6aa9a56ULL,
    0xaaaaaaaa9a5aULL,
    0xaaabaabaaeaaULL,
    0x569665a65aaaULL,
    0x5aa6a6a69a56ULL,
    0x6aaaaaaa9a5aULL,
    0xaaabaabaaeaaULL,
    0x569665a65aaaULL,
    0x5aa6a6a65a56ULL,
    0x6aaaaaaa9a5aULL,
    0xaaabaabaaa6aULL,
    0x569665a65aaaULL,
    0x5aa6a6a65a56ULL,
    0x6aaaa6aa9a5aULL,
    0xaaaaaabaaa6aULL,
    0x555665665aaaULL,
    0x5aa665a65a56ULL,
    0x6aaaa6aa9a5aULL,
    0xaaaaaabaaa6aULL,
    0x555665665aaaULL,
    0x5aa665a65a56ULL,
    0x6aaaa6aa9a5aULL,
    0xaaaaaaaaaa6aULL,
    0x555665665aaaULL,
    0x5aa665a65a56ULL,
    0x6aaaa6aa9a5aULL,
    0xaaaaaaaaaa6aULL,
    0x555665665aaaULL,
    0x5aa665a65a56ULL,
    0x6aaaa6aa9a5aULL,
    0xaaaaaaaaaa6aULL,
    0x555665655aaaULL,
    0x569665a65a56ULL,
    0x6aa6a6aa9a56ULL,
    0xaaaaaaaa9a5aULL,
    0x5556556559aaULL,
    0x569665a65a55ULL,
    0x6aa6a6a65a56ULL,
    0xaaaaaaaa9a5aULL,
    0x5556556559aaULL,
    0x569665a65a55ULL,
    0x5aa6a6a65a56ULL,
    0x6aaaa6aa9a5aULL,
    0x5556556555aaULL,
    0x569665a65a55ULL,
    0x5aa665a65a56ULL,
    0x6aaaa6aa9a5aULL,
    0x55555565556aULL,
    0x555665665a55ULL,
    0x5aa665a65a56ULL,
    0x6aaaa6aa9a5aULL,
    0x55555565556aULL,
    0x555665665a55ULL,
    0x5aa665a65a56ULL,
    0x6aaaa6aa9a5aULL,
    0x55555555556aULL,
    0x555665665a55ULL,
    0x5aa665a65a56ULL,
    0x6aaaa6aa9a5aULL,
    0x55555555556aULL,
    0x555665655a55ULL,
    0x5aa665a65a56ULL,
    0x6aa6a6aa9a5aULL,
    0x55555555456aULL,
    0x555655655a55ULL,
    0x5a9665a65a56ULL,
    0x6aa6a6a69a5aULL,
    0x55555555456aULL,
    0x555655655a55ULL,
    0x569665a65a56ULL,
    0x6aa6a6a65a56ULL,
    0x55555155455aULL,
    0x555655655955ULL,
    0x569665a65a55ULL,
    0x5aa6a5a65a56ULL,
    0x15555155455aULL,
    0x555555655555ULL,
    0x569665665a55ULL,
    0x5aa665a65a56ULL,
    0x15555155455aULL,
    0x555555655515ULL,
    0x555665665a55ULL,
    0x5aa665a65a56ULL,
    0x15555155455aULL,
    0x555555555515ULL,
    0x555665665a55ULL,
    0x5aa665a65a56ULL,
    0x15555155455aULL,
    0x555555555515ULL,
    0x555665665a55ULL,
    0x5aa665a65a56ULL,
    0x15555155455aULL,
    0x555555555515ULL,
    0x555655655a55ULL,
    0x5aa665a65a56ULL,
    0x15515155455aULL,
    0x555555554515ULL,
    0x555655655a55ULL,
    0x5a9665a65a56ULL,
    0x15515151455aULL,
    0x555551554515ULL,
    0x555655655a55ULL,
    0x569665a65a56ULL,
    0x155151510556ULL,
    0x555551554505ULL,
    0x555655655955ULL,
    0x569665665a55ULL,
    0x155110510556ULL,
    0x155551554505ULL,
    0x555555655555ULL,
    0x569665665a55ULL,
    0x55110510556ULL,
    0x155551554505ULL,
    0x555555555515ULL,
    0x555665665a55ULL,
    0x55110510556ULL,
    0x155551554505ULL,
    0x555555555515ULL,
    0x555665665a55ULL,
    0x55110510556ULL,
    0x155551554505ULL,
    0x555555555515ULL,
    0x555655655a55ULL,
    0x55110510556ULL,
    0x155551554505ULL,
    0x555555555515ULL,
    0x555655655a55ULL,
    0x55110510556ULL,
    0x155151514505ULL,
    0x555555554515ULL,
    0x555655655a55ULL,
    0x54110510556ULL,
    0x155151510505ULL,
    0x555551554515ULL,
    0x555655655a55ULL,
    0x14110110556ULL,
    0x155110510501ULL,
    0x555551554505ULL,
    0x555555655555ULL,
    0x14110110555ULL,
    0x155110510501ULL,
    0x555551554505ULL,
    0x555555555555ULL,
    0x14110110555ULL,
    0x55110510501ULL,
    0x155551554505ULL,
    0x555555555555ULL,
    0x110110555ULL,
    0x55110510501ULL,
    0x155551554505ULL,
    0x555555555515ULL,
    0x110110555ULL,
    0x55110510501ULL,
    0x155551554505ULL,
    0x555555555515ULL,
    0x100100555ULL,
    0x55110510501ULL,
    0x155151514505ULL,
    0x555555555515ULL,
    0x100100555ULL,
    0x54110510501ULL,
    0x155151514505ULL,
    0x555551554515ULL,
    0x100100555ULL,
    0x54110510501ULL,
    0x155150510505ULL,
    0x555551554515ULL,
    0x100100555ULL,
    0x14110110501ULL,
    0x155110510505ULL,
    0x555551554505ULL,
    0x100055ULL,
    0x14110110500ULL,
    0x155110510501ULL,
    0x555551554505ULL,
    0x55ULL,
    0x14110110500ULL,
    0x55110510501ULL,
    0x155551554505ULL,
    0x55ULL,
    0x110110500ULL,
    0x55110510501ULL,
    0x155551554505ULL,
    0x15ULL,
    0x100110500ULL,
    0x55110510501ULL,
    0x155551554505ULL,
    0x555555555515ULL,
};

static const unsigned int cnl_LUNAR_MONTH[199] = {
    0x752,
    0xea5,
    0xab2a,
    0x64b,
    0xa9b,
    0x9aa6,
    0x56a,
    0xb59,
    0x4baa,
    0x752,
    0xcda5,
    0xb25,
    0xa4b,
    0xba4b,
    0x2ad,
    0x56b,
    0x45b5,
    0xda9,
    0xfe92,
    0xe92,
    0xd25,
    0xad2d,
    0xa56,
    0x2b6,
    0x9ad5,
    0x6d4,
    0xea9,
    0x4f4a,
    0xe92,
    0xc6a6,
    0x52b,
    0xa57,
    0xb956,
    0xb5a,
    0x6d4,
    0x7761,
    0x749,
    0xfb13,
    0xa93,
    0x52b,
    0xd51b,
    0xaad,
    0x56a,
    0x9da5,
    0xba4,
    0xb49,
    0x4d4b,
    0xa95,
    0xeaad,
    0x536,
    0xaad,
    0xbaca,
    0x5b2,
    0xda5,
    0x7ea2,
    0xd4a,
    0x10595,
    0xa97,
    0x556,
    0xc575,
    0xad5,
    0x6d2,
    0x8755,
    0xea5,
    0x64a,
    0x664f,
    0xa9b,
    0xeada,
    0x56a,
    0xb69,
    0xabb2,
    0xb52,
    0xb25,
    0x8b2b,
    0xa4b,
    0x10aab,
    0x2ad,
    0x56d,
    0xd5a9,
    0xda9,
    0xd92,
    0x8e95,
    0xd25,
    0x14e4d,
    0xa56,
    0x2b6,
    0xc2f5,
    0x6d5,
    0xea9,
    0xaf52,
    0xe92,
    0xd26,
    0x652e,
    0xa57,
    0x10ad6,
    0x35a,
    0x6d5,
    0xab69,
    0x749,
    0x693,
    0x8a9b,
    0x52b,
    0xa5b,
    0x4aae,
    0x56a,
    0xedd5,
    0xba4,
    0xb49,
    0xad53,
    0xa95,
    0x52d,
    0x855d,
    0xab5,
    0x12baa,
    0x5d2,
    0xda5,
    0xde8a,
    0xd4a,
    0xc95,
    0x8a9e,
    0x556,
    0xab5,
    0x4ada,
    0x6d2,
    0xc765,
    0x725,
    0x64b,
    0xa657,
    0xcab,
    0x55a,
    0x656e,
    0xb69,
    0x16f52,
    0xb52,
    0xb25,
    0xdd0b,
    0xa4b,
    0x4ab,
    0xa2bb,
    0x5ad,
    0xb6a,
    0x4daa,
    0xd92,
    0xeea5,
    0xd25,
    0xa55,
    0xba4d,
    0x4b6,
    0x5b5,
    0x76d2,
    0xec9,
    0x10f92,
    0xe92,
    0xd26,
    0xd516,
    0xa57,
    0x556,
    0x9365,
    0x755,
    0x749,
    0x674b,
    0x693,
    0xeaab,
    0x52b,
    0xa5b,
    0xaaba,
    0x56a,
    0xb65,
    0x8baa,
    0xb4a,
    0x10d95,
    0xa95,
    0x52d,
    0xc56d,
    0xab5,
    0x5aa,
    0x85d5,
    0xda5,
    0xd4a,
    0x6e4d,
    0xc96,
    0xecce,
    0x556,
    0xab5,
    0xbad2,
    0x6d2,
    0xea5,
    0x872a,
    0x68b,
    0x10697,
    0x4ab,
    0x55b,
    0xd556,
    0xb6a,
    0x752,
    0x8b95,
    0xb45,
    0xa8b,
    0x4a4f,
};

static const unsigned int cnl_NEWYEAR[200] = {
    0x53,
    0x48,
    0x3d,
    0x50,
    0x44,
    0x39,
    0x4d,
    0x42,
    0x36,
    0x4a,
    0x3e,
    0x52,
    0x46,
    0x3a,
    0x4e,
    0x43,
    0x37,
    0x4b,
    0x41,
    0x54,
    0x48,
    0x3c,
    0x50,
    0x45,
    0x38,
    0x4d,
    0x42,
    0x37,
    0x4a,
    0x3e,
    0x51,
    0x46,
    0x3a,
    0x4e,
    0x44,
    0x38,
    0x4b,
    0x3f,
    0x53,
    0x48,
    0x3b,
    0x4f,
    0x45,
    0x39,
    0x4d,
    0x42,
    0x36,
    0x4a,
    0x3d,
    0x51,
    0x46,
    0x3b,
    0x4e,
    0x43,
    0x38,
    0x4c,
    0x3f,
    0x52,
    0x48,
    0x3c,
    0x4f,
    0x45,
    0x39,
    0x4d,
    0x42,
    0x35,
    0x49,
    0x3e,
    0x51,
    0x46,
    0x3b,
    0x4f,
    0x43,
    0x37,
    0x4b,
    0x3f,
    0x52,
    0x47,
    0x3c,
    0x50,
    0x45,
    0x39,
    0x4d,
    0x42,
    0x54,
    0x49,
    0x3d,
    0x51,
    0x46,
    0x3b,
    0x4f,
    0x44,
    0x37,
    0x4a,
    0x3f,
    0x53,
    0x47,
    0x3c,
    0x50,
    0x45,
    0x38,
    0x4c,
    0x41,
    0x36,
    0x49,
    0x3d,
    0x52,
    0x47,
    0x3a,
    0x4e,
    0x43,
    0x37,
    0x4a,
    0x3f,
    0x53,
    0x48,
    0x3c,
    0x50,
    0x45,
    0x39,
    0x4c,
    0x41,
    0x36,
    0x4a,
    0x3d,
    0x51,
    0x46,
    0x3a,
    0x4d,
    0x43,
    0x37,
    0x4b,
    0x3f,
    0x53,
    0x48,
    0x3c,
    0x4f,
    0x44,
    0x38,
    0x4c,
    0x41,
    0x36,
    0x4a,
    0x3e,
    0x51,
    0x46,
    0x3a,
    0x4e,
    0x42,
    0x37,
    0x4b,
    0x41,
    0x53,
    0x48,
    0x3c,
    0x4f,
    0x44,
    0x38,
    0x4c,
    0x42,
    0x35,
    0x49,
    0x3d,
    0x51,
    0x45,
    0x3a,
    0x4e,
    0x43,
    0x37,
    0x4b,
    0x3f,
    0x53,
    0x47,
    0x3b,
    0x4f,
    0x45,
    0x38,
    0x4c,
    0x42,
    0x36,
    0x49,
    0x3d,
    0x51,
    0x46,
    0x3a,
    0x4e,
    0x43,
    0x38,
    0x4a,
    0x3e,
    0x52,
    0x47,
    0x3b,
    0x4f,
    0x45,
    0x39,
    0x4c,
    0x41,
    0x35,
    0x49,
};

static const unsigned short cnl_TWOHOUR_LUCKY[60] = {
    0x2d3,
    0xcb4,
    0x32d,
    0x4cb,
    0xd32,
    0xb4c,
    0x2d3,
    0xcb4,
    0x32d,
    0x4cb,
    0xd22,
    0xb5c,
    0x2d3,
    0xcb4,
    0x32d,
    0x4cb,
    0xd3a,
    0xb4d,
    0x2d3,
    0xcb4,
    0x32d,
    0x4cb,
    0xd32,
    0xb4c,
    0x2d3,
    0xcb5,
    0x32d,
    0x4cb,
    0xd32,
    0xb4c,
    0x2d3,
    0xcb4,
    0x32d,
    0x4cb,
    0xd32,
    0xb4c,
    0x2d3,
    0xcb4,
    0x32d,
    0x4db,
    0xd32,
    0xb5c,
    0x2d7,
    0xcb4,
    0x32d,
    0x4cb,
    0xd32,
    0xb5c,
    0x2d3,
    0xcb4,
    0x32d,
    0x4cb,
    0xd32,
    0xb4c,
    0x2d3,
    0xcb4,
    0x30d,
    0x4cb,
    0xd32,
    0xb4c,
};

static const char *const cnl_THINGS_SORT[] = {
    "祭祀",
    "出行",
    "移徙",
    "结婚姻",
    "宴会",
    "嫁娶",
    "安床",
    "沐浴",
    "剃头",
    "修造",
    "求医疗病",
    "上表章",
    "上官",
    "入学",
    "冠带",
    "进人口",
    "裁衣",
    "竖柱上梁",
    "经络",
    "开市",
    "立券",
    "交易",
    "纳财",
    "修置产室",
    "开渠",
    "穿井",
    "安碓硙",
    "扫舍宇",
    "平治道涂",
    "破屋坏垣",
    "伐木",
    "捕捉",
    "畋猎",
    "栽种",
    "牧养",
    "破土",
    "安葬",
    "启攒",
};

static const char *const cnl_FETAL_GOD[] = {
    "碓磨门外东南",
    "碓磨厕外东南",
    "厨灶炉外正南",
    "仓库门外正南",
    "房床厕外正南",
    "占门床外正南",
    "占碓磨外正南",
    "厨灶厕外西南",
    "仓库炉外西南",
    "房床门外西南",
    "门碓栖外西南",
    "碓磨床外西南",
    "厨灶碓外西南",
    "仓库厕外西南",
    "房床厕外正南",
    "房床炉外正西",
    "碓磨栖外正西",
    "厨灶床外正西",
    "仓库碓外西北",
    "房床厕外西北",
    "占门炉外西北",
    "碓磨门外西北",
    "厨灶栖外西北",
    "仓库床外西北",
    "房床碓外正北",
    "占门厕外正北",
    "碓磨炉外正北",
    "厨灶门外正北",
    "仓库栖外正北",
    "占房床房内北",
    "占门碓房内北",
    "碓磨门房内北",
    "厨灶炉房内北",
    "仓库门房内北",
    "房床栖房内中",
    "占门床房内中",
    "占碓磨房内南",
    "厨灶厕房内南",
    "仓库炉房内南",
    "房床门房内南",
    "门鸡栖房内东",
    "碓磨床房内东",
    "厨灶碓房内东",
    "仓库厕房内东",
    "房床炉房内东",
    "占大门外东北",
    "碓磨栖外东北",
    "厨灶床外东北",
    "仓库碓外东北",
    "房床厕外东北",
    "占门炉外东北",
    "碓磨门外正东",
    "厨灶栖外正东",
    "仓库床外正东",
    "房床碓外正东",
    "占门厕外正东",
    "碓磨炉外东南",
    "仓库栖外东南",
    "占房床外东南",
    "占门碓外东南",
};

static const char *const cnl_MERIDIANS[] = {
    "胆",
    "肝",
    "肺",
    "大肠",
    "胃",
    "脾",
    "心",
    "小肠",
    "膀胱",
    "肾",
    "心包",
    "三焦",
};

/* 子午流注 说明/宜/忌（与 almanac.js calculateZiWuLiuZhu 一致，按子..亥索引） */
static const char *const cnl_MERIDIAN_NOTE[] = {
    "子时睡得足，黑眼圈不露。子时前入睡，晨醒后头脑清新，气色红润。",
    "人卧则血归于肝。此时应熟睡养肝，让肝血推陈出新，顺利排毒藏血。",
    "寅时睡得熟，面红精气足。此时肺经最旺，熟睡可保证气血平均分配。",
    "卯时大肠蠕，排毒渣滓出。此时宜披衣起床。",
    "辰时吃早餐，营养身体安。早餐宜丰富及多样化。",
    "巳时脾经旺，造血身体壮。此时宜适当活动，忌久坐不动。",
    "午时一小憩，安神养精气。午时小睡片刻或闭目养神，有助于养心。",
    "未时分清浊，饮水能降火。此时宜多喝水、喝茶，利于小肠排毒降火。",
    "申时津液足，养阴身体舒。此时宜运动，有助于体内津液循环。",
    "日出而作，日入而息。此时适宜下班。",
    "戌时护心脏，减压心舒畅。此时宜保持心情舒畅。",
    "亥时百脉通，养身养娇容。此时宜睡眠，休息百脉，有益美容。",
};
static const char *const cnl_MERIDIAN_YI[] = {
    "睡觉",
    "熟睡",
    "熟睡，或导引吐纳、调理肺经",
    "起床喝温热的白开水，排便，调理大肠经",
    "及时吃早餐，调理胃经",
    "适量饮水，调理脾经",
    "吃午餐，小憩，静养阴血，调理心经",
    "调理小肠经",
    "适量饮水，运动，工作，调理膀胱经",
    "休息，调理肾经",
    "吃晚餐，减压，散步，调理心包经",
    "心平气和，入睡，调理三焦经",
};
static const char *const cnl_MERIDIAN_JI[] = {
    "熬夜、吃夜宵",
    "熬夜，生闷气、久视",
    "熬夜",
    "饮酒",
    "早餐质量不好",
    "思虑过度，久坐不动",
    "马上剧烈运动",
    "多吃食物",
    "憋小便",
    "过劳",
    "晚餐过肥腻，生气",
    "熬夜，生气，饮茶",
};

static const char *const cnl_BUJIANG[] = {
    "壬寅壬辰辛丑辛卯辛巳庚寅庚辰丁丑丁卯丁巳戊寅戊辰",
    "辛丑辛卯庚子庚寅庚辰丁丑丁卯丙子丙寅丙辰戊子戊寅戊辰",
    "辛亥辛丑辛卯庚子庚寅丁亥丁丑丁卯丙子丙寅戊子戊寅",
    "庚戌庚子庚寅丁亥丁丑丙戌丙子丙寅乙亥乙丑戊戌戊子戊寅",
    "丁酉丁亥丁丑丙戌丙子乙酉乙亥乙丑甲戌甲子戊戌戊子",
    "丁酉丁亥丙申丙戌丙子乙酉乙亥甲申甲戌甲子戊申戊戌戊子",
    "丙申丙戌乙未乙酉乙亥甲申甲戌癸未癸酉癸亥戊申戊戌",
    "乙未乙酉甲午甲申甲戌癸未癸酉壬午壬申壬戌戊午戊申戊戌",
    "乙巳乙未乙酉甲午甲申癸巳癸未癸酉壬午壬申戊午戊申",
    "甲辰甲午甲申癸巳癸未壬辰壬午壬申辛巳辛未戊辰戊午戊申",
    "癸卯癸巳癸未壬辰壬午辛卯辛巳辛未庚辰庚午戊辰戊午",
    "癸卯癸巳壬寅壬辰壬午辛卯辛巳庚寅庚辰庚午戊寅戊辰戊午",
};

static const char *const cnl_MONTH_TYPE = "\xe4\xbb\xb2\xe5\xad\xa3\xe5\xad\x9f";
static const char *const cnl_SEASON_TYPE = "\xe6\x98\xa5\xe5\xa4\x8f\xe7\xa7\x8b\xe5\x86\xac";
static const char *const cnl_STEM10_BASE = "甲乙丙丁戊己庚辛壬癸";
static const char *const cnl_BRANCH12_BASE = "子丑寅卯辰巳午未申酉戌亥";
static const char *const cnl_STEM5ELEM_BASE = "木木火火土土金金水水";
static const char *const cnl_BRANCH5ELEM_BASE = "水土木木土火火土金金土水";

/* 其他阳历假日 平铺表 + 每月的起止下标 */
typedef struct { int day; const char *name; } cnl_hday;
static const cnl_hday cnl_OTHER_HOLIDAY[] = {
    {8, "周恩来逝世纪念日"},
    {10, "中国公安110宣传日"},
    {21, "列宁逝世纪念日"},
    {26, "国际海关日"},
    {2, "世界湿地日"},
    {4, "世界抗癌日"},
    {7, "京汉铁路罢工纪念"},
    {10, "国际气象节"},
    {14, "情人节"},
    {19, "邓小平逝世纪念日"},
    {21, "国际母语日"},
    {24, "第三世界青年日"},
    {1, "国际海豹日"},
    {3, "全国爱耳日"},
    {5, "周恩来诞辰纪念日,中国青年志愿者服务日"},
    {6, "世界青光眼日"},
    {8, "国际劳动妇女节"},
    {12, "孙中山逝世纪念日,中国植树节"},
    {14, "马克思逝世纪念日"},
    {15, "国际消费者权益日"},
    {17, "国际航海日"},
    {18, "全国科技人才活动日"},
    {21, "世界森林日,世界睡眠日"},
    {22, "世界水日"},
    {23, "世界气象日"},
    {24, "世界防治结核病日"},
    {1, "国际愚人节"},
    {2, "国际儿童图书日"},
    {7, "世界卫生日"},
    {22, "列宁诞辰纪念日"},
    {23, "世界图书和版权日"},
    {26, "世界知识产权日"},
    {3, "世界新闻自由日"},
    {4, "中国青年节"},
    {5, "马克思诞辰纪念日"},
    {8, "世界红十字日"},
    {11, "世界肥胖日"},
    {23, "世界读书日"},
    {27, "上海解放日"},
    {31, "世界无烟日"},
    {1, "国际儿童节"},
    {5, "世界环境日"},
    {6, "全国爱眼日"},
    {8, "世界海洋日"},
    {11, "中国人口日"},
    {14, "世界献血日"},
    {1, "中国共产党诞生日,香港回归纪念日"},
    {7, "中国人民抗日战争纪念日"},
    {11, "世界人口日"},
    {1, "中国人民解放军建军节"},
    {5, "恩格斯逝世纪念日"},
    {6, "国际电影节"},
    {12, "国际青年日"},
    {22, "邓小平诞辰纪念日"},
    {3, "中国抗日战争胜利纪念日"},
    {8, "世界扫盲日"},
    {9, "毛泽东逝世纪念日"},
    {10, "中国教师节"},
    {14, "世界清洁地球日"},
    {18, "“九·一八”事变纪念日"},
    {20, "全国爱牙日"},
    {21, "国际和平日"},
    {27, "世界旅游日"},
    {4, "世界动物日"},
    {10, "辛亥革命纪念日"},
    {13, "中国少年先锋队诞辰日"},
    {25, "抗美援朝纪念日"},
    {12, "孙中山诞辰纪念日"},
    {28, "恩格斯诞辰纪念日"},
    {1, "世界艾滋病日"},
    {12, "西安事变纪念日"},
    {13, "南京大屠杀纪念日"},
    {24, "平安夜"},
    {25, "圣诞节"},
    {26, "毛泽东诞辰纪念日"},
};
static const int cnl_OTHER_HOLIDAY_IDX[13] = {0,4,12,26,32,40,46,49,54,63,67,69,75};

/* 其他农历假日 平铺表 + 每月起止下标 */
static const cnl_hday cnl_OTHER_LUNAR_HOLIDAY[] = {
    {1, "弥勒佛圣诞"},
    {8, "五殿阎罗天子诞"},
    {9, "玉皇上帝诞"},
    {15, "元宵节"},
    {1, "一殿秦广王诞"},
    {2, "春龙节-福德土地正神诞"},
    {3, "文昌帝君诞"},
    {6, "东华帝君诞"},
    {8, "释迦牟尼佛出家"},
    {15, "释迦牟尼佛般涅槃"},
    {17, "东方杜将军诞"},
    {18, "至圣先师孔子讳辰"},
    {19, "观音大士诞"},
    {21, "普贤菩萨诞"},
    {1, "二殿楚江王诞"},
    {3, "三月三-玄天上帝诞"},
    {8, "六殿卞城王诞"},
    {15, "昊天上帝诞"},
    {16, "准提菩萨诞"},
    {19, "中岳大帝诞"},
    {20, "子孙娘娘诞"},
    {27, "七殿泰山王诞"},
    {28, "苍颉至圣先师诞"},
    {1, "八殿都市王诞"},
    {4, "文殊菩萨诞"},
    {8, "释迦牟尼佛诞"},
    {14, "纯阳祖师诞"},
    {15, "钟离祖师诞"},
    {17, "十殿转轮王诞"},
    {18, "紫徽大帝诞"},
    {20, "眼光圣母诞"},
    {1, "南极长生大帝诞"},
    {8, "南方五道诞"},
    {11, "天下都城隍诞"},
    {12, "炳灵公诞"},
    {13, "关圣降"},
    {16, "天地元气造化万物之辰"},
    {18, "张天师诞"},
    {22, "孝娥神诞"},
    {19, "观世音菩萨成道日"},
    {24, "关帝诞"},
    {7, "七夕-魁星诞"},
    {13, "长真谭真人诞-大势至菩萨诞"},
    {15, "中元节"},
    {18, "西王母诞"},
    {19, "太岁诞"},
    {22, "增福财神诞"},
    {29, "杨公忌"},
    {30, "地藏菩萨诞"},
    {1, "许真君诞"},
    {3, "司命灶君诞"},
    {5, "雷声大帝诞"},
    {10, "北斗大帝诞"},
    {12, "西方五道诞"},
    {16, "天曹掠刷真君降"},
    {18, "天人兴福之辰"},
    {23, "汉恒候张显王诞"},
    {24, "灶君夫人诞"},
    {29, "至圣先师孔子诞"},
    {1, "北斗九星降世"},
    {3, "五瘟神诞"},
    {9, "重阳节-酆都大帝诞"},
    {13, "孟婆尊神诞"},
    {17, "金龙四大王诞"},
    {19, "观世音菩萨出家"},
    {30, "药师琉璃光佛诞"},
    {1, "寒衣节"},
    {3, "三茅诞"},
    {5, "达摩祖师诞"},
    {8, "佛涅槃日"},
    {15, "下元节"},
    {4, "至圣先师孔子诞"},
    {6, "西岳大帝诞"},
    {11, "太乙救苦天尊诞"},
    {17, "阿弥陀佛诞"},
    {19, "太阳日宫诞"},
    {23, "张仙诞"},
    {26, "北方五道诞"},
    {8, "腊八节-释迦如来成佛之辰"},
    {16, "南岳大帝诞"},
    {21, "天猷上帝诞"},
    {23, "小年"},
    {24, "子时灶君上天朝玉帝"},
    {29, "华严菩萨诞"},
    {30, "除夕"},
};
static const int cnl_OTHER_LUNAR_HOLIDAY_IDX[13] = {0,4,14,23,31,39,41,49,59,66,71,78,85};

/************* include *************/
#include <string.h>


/* ============================ PSRAM 安全原语 ============================ */

/* 预编译 libc 的 strcmp/strlen/memcmp/memcpy/memset 未带 ESP32 PSRAM 缓存修复
 * （反汇编确认无 memw 屏障）。对其上的 PSRAM 指针做字节扫描，会与 flash 指令/
 * 常量缓存填充竞争（ESP32 硅 bug），重负载下致 CPU 硬挂（实测：INT WDT 复位）。
 * 本项目其它模块（如 ui_dict 的 ci_cmp）的惯例是手写字节循环。下方自实现循环
 * 版本随本编译单元打上 -mfix-esp32-psram-cache-issue（memw），可安全访问 PSRAM；
 * 随后以宏将库函数名替换为本实现（仅本文件实现段内生效）。
 * 注意：必须定义于任何字符串/内存函数使用点之前，且本原语内部不得再调用库同名函数。 */
static size_t cnl_strlen(const char *s) { size_t n = 0; while (s[n]) ++n; return n; }

static int cnl_strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { ++a; ++b; }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

static int cnl_memcmp(const void *va, const void *vb, size_t n) {
    const unsigned char *a = (const unsigned char *)va;
    const unsigned char *b = (const unsigned char *)vb;
    for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) return (int)a[i] - (int)b[i];
    return 0;
}

static void cnl_memcpy(void *vd, const void *vs, size_t n) {
    unsigned char *d = (unsigned char *)vd;
    const unsigned char *s = (const unsigned char *)vs;
    for (size_t i = 0; i < n; ++i) d[i] = s[i];
}

/* 清零/填充（走 32 位字以加速 145KB 结果缓冲；字节收尾） */
static void cnl_memset(void *vd, int c, size_t n) {
    unsigned char *d = (unsigned char *)vd;
    unsigned int vv = (unsigned int)(unsigned char)c * 0x01010101u;
    while (n && (((unsigned long)d) & 3u)) { *d++ = (unsigned char)vv; --n; }
    while (n >= 4) { *(unsigned int *)d = vv; d += 4; n -= 4; }
    while (n) { *d++ = (unsigned char)vv; --n; }
}

static char *cnl_strstr(const char *h, const char *n) {
    size_t hl = 0, nl = 0;
    while (h[hl]) ++hl;
    while (n[nl]) ++nl;
    if (nl == 0) return (char *)h;
    if (nl > hl) return NULL;
    for (size_t i = 0; i + nl <= hl; ++i) {
        size_t k = 0;
        while (k < nl && h[i + k] == n[k]) ++k;
        if (k == nl) return (char *)(h + i);
    }
    return NULL;
}

static char *cnl_strncat(char *d, const char *s, size_t n) {
    char *p = d;
    while (*p) ++p;
    while (n-- > 0 && *s) *p++ = *s++;
    *p = 0;
    return d;
}

#define strcmp  cnl_strcmp
#define memcmp  cnl_memcmp
#define memcpy  cnl_memcpy
#define memset  cnl_memset
#define strlen  cnl_strlen
#define strstr  cnl_strstr
#define strncat cnl_strncat

/* ============================ HELPERS ============================ */

/* 嵌入式工作区（workspace）分配器：计算过程中的瞬态 cnlunar_list（每个 16KB）
 * 从调用方提供的工作区切取，避免 12KB 任务栈溢出。两条入口路径均保证 cs 非空：
 * cnlunar_calculate_ws 用调用方 workspace；cnlunar_calculate（宿主机便捷路径）
 * 在栈上自备 CNLUNAR_WORKSPACE_MIN 工作区（桌面栈足够大）。 */
typedef struct {
    void *base;      /* 工作区起始 */
    void *pos;       /* 当前切取位置 */
    void *end;       /* 工作区结束 */
} cnl_scratch;

static void *cnl_ws_alloc(cnl_scratch *cs, size_t n) {
    if (!cs) return NULL;
    /* 4 字节对齐 */
    n = (n + 3u) & ~(size_t)3u;
    if ((size_t)((char *)cs->end - (char *)cs->pos) < n) return NULL;
    void *p = cs->pos;
    cs->pos = (char *)cs->pos + n;
    return p;
}

/* 从工作区切取一个 cnlunar_list。注意：不得在此声明栈上回退变量——
   cnlunar_list 约 16KB，Core0 渲染任务栈仅 12KB，回退变量即使未被使用也
   占栈帧（多槽重叠时 ≥32KB），必然栈溢出引发 WDT 复位。容量由
   CNLUNAR_WORKSPACE_MIN 保证（cnl_holidays 用 1 槽、cnl_angel_demon 峰值
   3 槽：allgods + dibt + rm0/1/2 之一），且入口已校验 workspace 大小，
   分配失败理论上不可达；兜底 return 仅为防御。 */
#define CNL_WS_LIST(cs, var) \
    cnlunar_list *var = (cnlunar_list *)cnl_ws_alloc((cs), sizeof(cnlunar_list)); \
    if (!var) return; \
    cnl_l_reset(var);

static int cnl_mod(int a, int n) { int r = a % n; return r < 0 ? r + n : r; }

/* days since 1970-01-01 (proleptic Gregorian), C99 port of Hinnant's algorithm */
static int cnl_daynum(int y, int m, int d) {
    y -= m <= 2;
    int era = (y >= 0 ? y : y - 399) / 400;
    unsigned yoe = (unsigned)(y - era * 400);
    unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + (unsigned)(d - 1);
    unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (int)doe - 719468;
}

static int cnl_is_leap_year(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

static const int cnl_mdays_noleap[13] = {0,31,28,31,30,31,30,31,31,30,31,30,31};

/* decompose daynum back to y/m/d */
static void cnl_ymd(int days, int *y, int *m, int *d) {
    days += 719468;
    int era = (days >= 0 ? days : days - 146096) / 146097;
    unsigned doe = (unsigned)(days - era * 146097);
    unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int yy = (int)yoe + era * 400;
    unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    unsigned mp = (5 * doy + 2) / 153;
    int dd = (int)(doy - (153 * mp + 2) / 5 + 1);
    int mm = (int)(mp + (mp < 10 ? 3 : -9));
    yy += (mm <= 2);
    *y = yy; *m = mm; *d = dd;
}

/* day-of-year (1-based) for y/m/d */
static int cnl_doy(int y, int m, int d) {
    int j = 0;
    for (int i = 1; i < m; ++i) j += cnl_mdays_noleap[i];
    if (m > 2 && cnl_is_leap_year(y)) j += 1;
    return j + d;
}

/* ISO calendar for a daynum: (year, week 1-53, weekday 1-7 Mon=1) */
static void cnl_iso(int days, int *iso_year, int *iso_week, int *iso_wday) {
    int wd0 = cnl_mod(days + 3, 7);           /* Mon=0 */
    int thu = days + 3 - wd0;                  /* Thursday of this week */
    int ty, tm, td;
    cnl_ymd(thu, &ty, &tm, &td);
    *iso_year = ty;
    *iso_week = (cnl_doy(ty, tm, td) - 1) / 7 + 1;
    *iso_wday = wd0 + 1;
}

/* copy src into dst, bounded, NUL-terminated */
static void cnl_cpy(char *dst, const char *src, int max) {
    int i = 0;
    if (max <= 0) { if (max == 0) return; }
    while (i < max - 1 && src[i]) { dst[i] = src[i]; ++i; }
    dst[i] = 0;
}

/* copy the first `nchars` codepoints of src into dst, NUL-terminated */
static void cnl_utf8_take(char *dst, const char *src, int nchars) {
    int cur = 0;
    while (*src) {
        if (((*(const unsigned char *)src) & 0xc0) != 0x80) {  /* lead byte */
            if (cur >= nchars) break;
            ++cur;
        }
        *dst++ = *src++;
    }
    *dst = 0;
}

/* does NUL-free pattern t (tl bytes) appear inside s (sl bytes)? */
static int cnl_bytes_in(const char *s, int sl, const char *t, int tl) {
    if (tl == 0) return 1;
    if (tl > sl) return 0;
    for (int i = 0; i + tl <= sl; ++i) {
        int j;
        for (j = 0; j < tl; ++j) if (s[i + j] != t[j]) break;
        if (j == tl) return 1;
    }
    return 0;
}

/* ---- list ops (set semantics, preserve insertion order) ---- */
static void cnl_l_reset(cnlunar_list *L) { L->n = 0; }

static int cnl_l_has(const cnlunar_list *L, const char *s) {
    for (int i = 0; i < L->n; ++i) if (strcmp(L->it[i], s) == 0) return 1;
    return 0;
}
/* append if room (no dedup) */
static void cnl_l_push(cnlunar_list *L, const char *s) {
    if (L->n >= CNLUNAR_LIST_MAX) return;
    cnl_cpy(L->it[L->n], s, CNLUNAR_ITEM_MAX);
    L->n++;
}
/* append if room and absent */
static void cnl_l_add(cnlunar_list *L, const char *s) {
    if (cnl_l_has(L, s)) return;
    if (L->n >= CNLUNAR_LIST_MAX) return;
    cnl_cpy(L->it[L->n], s, CNLUNAR_ITEM_MAX);
    L->n++;
}
/* set-dedup preserving first occurrence (approximates Python set()) */
static void cnl_l_dedup(cnlunar_list *L) {
    int w = 0;
    for (int i = 0; i < L->n; ++i) {
        int dup = 0;
        for (int j = 0; j < w; ++j) if (strcmp(L->it[j], L->it[i]) == 0) { dup = 1; break; }
        if (dup) continue;
        if (w != i) cnl_cpy(L->it[w], L->it[i], CNLUNAR_ITEM_MAX);
        w++;
    }
    L->n = w;
}
/* remove all occurrences of s */
static void cnl_l_rm(cnlunar_list *L, const char *s) {
    int w = 0;
    for (int i = 0; i < L->n; ++i) {
        if (strcmp(L->it[i], s) == 0) continue;
        if (w != i) cnl_cpy(L->it[w], L->it[i], CNLUNAR_ITEM_MAX);
        w++;
    }
    L->n = w;
}
static void cnl_l_rm_arr(cnlunar_list *L, const char *const *a, int n) {
    for (int i = 0; i < n; ++i) cnl_l_rm(L, a[i]);
}
/* union with array — set semantics (rfAdd) */
static void cnl_l_union_arr(cnlunar_list *L, const char *const *a, int n) {
    for (int i = 0; i < n; ++i) cnl_l_add(L, a[i]);
}
/* union from a cnlunar_list (list-to-list, row array) */
static void cnl_l_union_from(cnlunar_list *L, const cnlunar_list *src) {
    for (int i = 0; i < src->n; ++i) cnl_l_add(L, src->it[i]);
}
/* remove every item of src from L */
static void cnl_l_rm_list(cnlunar_list *L, const cnlunar_list *src) {
    for (int i = 0; i < src->n; ++i) cnl_l_rm(L, src->it[i]);
}
/* stable sort by thingsSort key (Python sort(key=sortCollation)) */
static int cnl_things_key(const char *s) {
    for (int i = 0; i < 38; ++i) if (strcmp(cnl_THINGS_SORT[i], s) == 0) return i;
    return 38;
}
static void cnl_l_sort_stable(cnlunar_list *L) {
    if (L->n <= 1) return;
    /* 原地稳定插入排序：仅用 64B 行缓冲。旧实现用 idx[]+tmp_[][] 暂存
       （栈上 17KB），会撑爆 12KB 任务栈。 */
    char tmp[CNLUNAR_ITEM_MAX];
    for (int i = 1; i < L->n; ++i) {
        int k = cnl_things_key(L->it[i]);
        memcpy(tmp, L->it[i], CNLUNAR_ITEM_MAX);
        int j = i;
        while (j > 0 && cnl_things_key(L->it[j - 1]) > k) {  /* 严格大于才后移：稳定 */
            memcpy(L->it[j], L->it[j - 1], CNLUNAR_ITEM_MAX);
            --j;
        }
        memcpy(L->it[j], tmp, CNLUNAR_ITEM_MAX);
    }
}

/* ---------- 神煞 row types (used by generated tables below) ---------- */
enum {
    CNL_OP_NONE = 0, CNL_OP_CIN = 1, CNL_OP_CSTEM = 2, CNL_OP_CBR = 3, CNL_OP_STAR0 = 4,
    CNL_OP_DSTEMGRP = 5, CNL_OP_DBRGRP = 6, CNL_OP_DBRSTR = 7, CNL_OP_DLIST = 8,
    CNL_OP_DLISTGRP = 9, CNL_OP_LDN = 10, CNL_OP_TMD = 11, CNL_OP_LMNLD = 12,
    CNL_OP_LMNCH = 13, CNL_OP_DSLICE = 14, CNL_OP_DBRSLICE = 15, CNL_OP_TUIDI = 16,
    CNL_OP_B3HE = 17, CNL_OP_BSUIP = 18, CNL_OP_BTIAN = 19, CNL_OP_TIANDE = 20
};
typedef struct {
    const char *name;
    int op;
    const char *base;
    int arg;
    const char *const *list;
    int list_n;
    const int *pairs;
    const char *const *pstr;
    int pairs_n;
    const char *const *good;
    const char *const *bad;
    int n_good, n_bad;
} cnl_god_row;
static const char *const const cnl_noarr[1] = { "" };

static const char *const ll000[1] = {
    "\xe4\xbf\xae\xe9\x80\xa0",
};
static const char *const ll001[37] = {
    "\xe7\xa5\xad\xe7\xa5\x80",
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe8\xa6\x83\xe6\x81\xa9",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe6\x81\xa4\xe5\xad\xa4\xe8\x8c\x95",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe9\x9b\xaa\xe5\x86\xa4",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
    "\xe5\xae\x89\xe8\x91\xac",
};
static const char *const ll002[2] = {
    "\xe7\x95\x8b\xe7\x8c\x8e",
    "\xe5\x8f\x96\xe9\xb1\xbc",
};
static const char *const ll003[12] = {
    "\xe5\xb7\xb3\xe8\xbe\xb0",
    "\xe5\xba\x9a",
    "\xe4\xb8\x81",
    "\xe7\x94\xb3\xe6\x9c\xaa",
    "\xe5\xa3\xac",
    "\xe8\xbe\x9b",
    "\xe4\xba\xa5\xe6\x88\x8c",
    "\xe7\x94\xb2",
    "\xe7\x99\xb8",
    "\xe5\xaf\x85\xe4\xb8\x91",
    "\xe4\xb8\x99",
    "\xe4\xb9\x99",
};
static const char *const ll004[1] = {
    "\xe5\xab\x81\xe5\xa8\xb6",
};
static const char *const ll005[18] = {
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe7\xbb\x8f\xe7\xbb\x9c",
    "\xe9\x85\x9d\xe9\x85\xbf",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xae\x89\xe7\xa2\x93\xe7\xa1\x99",
    "\xe7\xba\xb3\xe7\x95\x9c",
};
static const char *const ll006[24] = {
    "\xe7\xa5\xad\xe7\xa5\x80",
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
};
static const char *const ll007[4] = {
    "\xe4\xb8\x99\xe4\xb8\x81",
    "\xe6\x88\x8a\xe5\xb7\xb1",
    "\xe5\xa3\xac\xe7\x99\xb8",
    "\xe7\x94\xb2\xe4\xb9\x99",
};
static const char *const ll008[3] = {
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
};
static const char *const ll009[9] = {
    "\xe7\xbb\x8f\xe7\xbb\x9c",
    "\xe9\x85\x9d\xe9\x85\xbf",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
};
static const char *const ll010[10] = {
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe7\xbb\x8f\xe7\xbb\x9c",
    "\xe9\x85\x9d\xe9\x85\xbf",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe7\xba\xb3\xe7\x95\x9c",
    "\xe5\xae\x89\xe8\x91\xac",
};
static const char *const ll011[1] = {
    "\xe4\xb8\xb4\xe6\x94\xbf",
};
static const char *const ll012[1] = {
    "\xe5\xae\x89\xe8\x91\xac",
};
static const char *const ll013[12] = {
    "\xe5\xa3\xac\xe7\x94\xb3",
    "\xe7\x99\xb8\xe9\x85\x89",
    "\xe5\xa3\xac\xe5\x8d\x88",
    "\xe7\x94\xb2\xe7\x94\xb3",
    "\xe4\xb9\x99\xe9\x85\x89",
    "\xe4\xb8\x99\xe7\x94\xb3",
    "\xe4\xb8\x81\xe9\x85\x89",
    "\xe5\xa3\xac\xe5\xaf\x85",
    "\xe4\xb8\x99\xe5\x8d\x88",
    "\xe5\xb7\xb1\xe9\x85\x89",
    "\xe5\xba\x9a\xe7\x94\xb3",
    "\xe8\xbe\x9b\xe9\x85\x89",
};
static const char *const ll014[2] = {
    "\xe7\xa0\xb4\xe5\x9c\x9f",
    "\xe5\xae\x89\xe8\x91\xac",
};
static const char *const ll015[14] = {
    "\xe5\xba\x9a\xe5\x8d\x88",
    "\xe5\xa3\xac\xe7\x94\xb3",
    "\xe7\x99\xb8\xe9\x85\x89",
    "\xe5\xa3\xac\xe5\x8d\x88",
    "\xe7\x94\xb2\xe7\x94\xb3",
    "\xe4\xb9\x99\xe9\x85\x89",
    "\xe5\xb7\xb1\xe9\x85\x89",
    "\xe4\xb8\x99\xe7\x94\xb3",
    "\xe4\xb8\x81\xe9\x85\x89",
    "\xe5\xa3\xac\xe5\xaf\x85",
    "\xe4\xb8\x99\xe5\x8d\x88",
    "\xe5\xba\x9a\xe5\xaf\x85",
    "\xe5\xba\x9a\xe7\x94\xb3",
    "\xe8\xbe\x9b\xe9\x85\x89",
};
static const char *const ll016[7] = {
    "\xe5\xba\x9a\xe5\x8d\x88",
    "\xe5\xa3\xac\xe8\xbe\xb0",
    "\xe7\x94\xb2\xe8\xbe\xb0",
    "\xe4\xb9\x99\xe5\xb7\xb3",
    "\xe7\x94\xb2\xe5\xaf\x85",
    "\xe4\xb8\x99\xe8\xbe\xb0",
    "\xe5\xba\x9a\xe5\xaf\x85",
};
static const char *const ll017[2] = {
    "\xe7\xa0\xb4\xe5\x9c\x9f",
    "\xe5\x90\xaf\xe6\x94\x92",
};
static const char *const ll018[10] = {
    "\xe4\xb8\x99\xe5\xaf\x85",
    "\xe4\xb8\x81\xe5\x8d\xaf",
    "\xe4\xb8\x99\xe5\xad\x90",
    "\xe8\xbe\x9b\xe5\x8d\xaf",
    "\xe7\x94\xb2\xe5\x8d\x88",
    "\xe5\xba\x9a\xe5\xad\x90",
    "\xe7\x99\xb8\xe5\x8d\xaf",
    "\xe5\xa3\xac\xe5\xad\x90",
    "\xe7\x94\xb2\xe5\xaf\x85",
    "\xe4\xb9\x99\xe5\x8d\xaf",
};
static const char *const ll019[1] = {
    "\xe7\xa0\xb4\xe5\x9c\x9f",
};
static const char *const ll020[22] = {
    "\xe5\xba\x9a\xe5\x8d\x88",
    "\xe8\xbe\x9b\xe6\x9c\xaa",
    "\xe5\xa3\xac\xe7\x94\xb3",
    "\xe7\x99\xb8\xe9\x85\x89",
    "\xe6\x88\x8a\xe5\xaf\x85",
    "\xe5\xb7\xb1\xe5\x8d\xaf",
    "\xe5\xa3\xac\xe5\x8d\x88",
    "\xe7\x99\xb8\xe6\x9c\xaa",
    "\xe7\x94\xb2\xe7\x94\xb3",
    "\xe4\xb9\x99\xe9\x85\x89",
    "\xe4\xb8\x81\xe6\x9c\xaa",
    "\xe7\x94\xb2\xe5\x8d\x88",
    "\xe4\xb9\x99\xe6\x9c\xaa",
    "\xe4\xb8\x99\xe7\x94\xb3",
    "\xe4\xb8\x81\xe9\x85\x89",
    "\xe5\xa3\xac\xe5\xaf\x85",
    "\xe7\x99\xb8\xe5\x8d\xaf",
    "\xe4\xb8\x99\xe5\x8d\x88",
    "\xe6\x88\x8a\xe7\x94\xb3",
    "\xe5\xb7\xb1\xe9\x85\x89",
    "\xe5\xba\x9a\xe7\x94\xb3",
    "\xe8\xbe\x9b\xe9\x85\x89",
};
static const char *const ll021[16] = {
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe8\xa6\x83\xe6\x81\xa9",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe6\x81\xa4\xe5\xad\xa4\xe8\x8c\x95",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe9\x9b\xaa\xe5\x86\xa4",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe8\xa3\x81\xe5\x88\xb6",
};
static const char *const ll022[2] = {
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
};
static const char *const ll023[3] = {
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
};
static const char *const ll024[11] = {
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
};
static const char *const ll025[4] = {
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
};
static const char *const ll026[4] = {
    "\xe7\x94\xb2\xe4\xb9\x99",
    "\xe4\xb8\x99\xe4\xb8\x81",
    "\xe5\xba\x9a\xe8\xbe\x9b",
    "\xe5\xa3\xac\xe7\x99\xb8",
};
static const char *const ll027[10] = {
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
};
static const char *const ll028[2] = {
    "\xe5\xae\x89\xe8\x91\xac",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
};
static const char *const ll029[6] = {
    "\xe8\xa6\x83\xe6\x81\xa9",
    "\xe6\x81\xa4\xe5\xad\xa4\xe8\x8c\x95",
    "\xe5\xb8\x83\xe6\x94\xbf\xe4\xba\x8b",
    "\xe9\x9b\xaa\xe5\x86\xa4",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
};
static const char *const ll030[36] = {
    "\xe7\xa5\xad\xe7\xa5\x80",
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe8\xa6\x83\xe6\x81\xa9",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe6\x81\xa4\xe5\xad\xa4\xe8\x8c\x95",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe9\x9b\xaa\xe5\x86\xa4",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
    "\xe5\xae\x89\xe8\x91\xac",
};
static const char *const ll031[12] = {
    "\xe7\x94\xb2\xe5\xad\x90",
    "\xe7\x94\xb2\xe5\xad\x90",
    "\xe6\x88\x8a\xe5\xaf\x85",
    "\xe6\x88\x8a\xe5\xaf\x85",
    "\xe6\x88\x8a\xe5\xaf\x85",
    "\xe7\x94\xb2\xe5\x8d\x88",
    "\xe7\x94\xb2\xe5\x8d\x88",
    "\xe7\x94\xb2\xe5\x8d\x88",
    "\xe6\x88\x8a\xe7\x94\xb3",
    "\xe6\x88\x8a\xe7\x94\xb3",
    "\xe6\x88\x8a\xe7\x94\xb3",
    "\xe7\x94\xb2\xe5\xad\x90",
};
static const char *const ll032[40] = {
    "\xe7\xa5\xad\xe7\xa5\x80",
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe8\xa6\x83\xe6\x81\xa9",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe6\x81\xa4\xe5\xad\xa4\xe8\x8c\x95",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe9\x9b\xaa\xe5\x86\xa4",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe7\xbb\x8f\xe7\xbb\x9c",
    "\xe9\x85\x9d\xe9\x85\xbf",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
    "\xe5\xae\x89\xe8\x91\xac",
};
static const char *const ll033[12] = {
    "\xe7\x94\xb2\xe5\xad\x90",
    "\xe7\x99\xb8\xe6\x9c\xaa",
    "\xe7\x94\xb2\xe5\x8d\x88",
    "\xe7\x94\xb2\xe6\x88\x8c",
    "\xe4\xb9\x99\xe9\x85\x89",
    "\xe4\xb8\x99\xe5\xad\x90",
    "\xe4\xb8\x81\xe4\xb8\x91",
    "\xe6\x88\x8a\xe5\x8d\x88",
    "\xe7\x94\xb2\xe5\xaf\x85",
    "\xe4\xb8\x99\xe8\xbe\xb0",
    "\xe8\xbe\x9b\xe5\x8d\xaf",
    "\xe6\x88\x8a\xe8\xbe\xb0",
};
static const char *const ll034[1] = {
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
};
static const char *const ll035[2] = {
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe6\x90\xac\xe7\xa7\xbb",
};
static const char *const ll036[2] = {
    "\xe7\xa5\xad\xe7\xa5\x80",
    "\xe7\xa5\x88\xe7\xa6\x8f",
};
static const char *const ll037[6] = {
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
};
static const char *const ll038[1] = {
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
};
static const char *const ll039[4] = {
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
};
static const char *const ll040[4] = {
    "\xe4\xba\xa5\xe5\xad\x90",
    "\xe5\xaf\x85\xe5\x8d\xaf",
    "\xe8\xbe\xb0\xe4\xb8\x91\xe6\x88\x8c\xe6\x9c\xaa",
    "\xe7\x94\xb3\xe9\x85\x89",
};
static const char *const ll041[3] = {
    "\xe8\xb5\xb4\xe4\xbb\xbb",
    "\xe8\xaf\x89\xe8\xae\xbc",
    "\xe5\xae\x89\xe8\x91\xac",
};
static const char *const ll042[1] = {
    "\xe7\xba\xb3\xe8\xb4\xa2",
};
static const char *const ll043[2] = {
    "\xe6\x81\xa4\xe5\xad\xa4\xe8\x8c\x95",
    "\xe9\x9b\xaa\xe5\x86\xa4",
};
static const char *const ll044[7] = {
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb2\x90\xe6\xb5\xb4",
    "\xe6\x95\xb4\xe5\xae\xb9",
    "\xe5\x89\x83\xe5\xa4\xb4",
    "\xe6\x95\xb4\xe6\x89\x8b\xe8\xb6\xb3\xe7\x94\xb2",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe6\x89\xab\xe8\x88\x8d\xe5\xae\x87",
};
static const char *const ll045[7] = {
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb2\x90\xe6\xb5\xb4",
    "\xe6\x95\xb4\xe5\xae\xb9",
    "\xe5\x89\x83\xe5\xa4\xb4",
    "\xe6\x95\xb4\xe6\x89\x8b\xe8\xb6\xb3\xe7\x94\xb2",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
};
static const char *const ll046[3] = {
    "\xe4\xbc\x90\xe6\x9c\xa8",
    "\xe7\x95\x8b\xe7\x8c\x8e",
    "\xe5\x8f\x96\xe9\xb1\xbc",
};
static const char *const ll047[3] = {
    "\xe7\xa5\xad\xe7\xa5\x80",
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
};
static const char *const ll048[3] = {
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe7\xba\xb3\xe7\x95\x9c",
};
static const char *const ll049[5] = {
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
};
static const char *const ll050[3] = {
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
};
static const char *const ll051[12] = {
    "\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3",
    "\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0",
    "\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf",
    "\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85",
    "\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91",
    "\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90",
    "\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5",
    "\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c",
    "\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89",
    "\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3",
    "\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa",
    "\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88",
};
static const char *const ll052[4] = {
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe5\x87\xba\xe8\xa1\x8c",
};
static const char *const ll053[10] = {
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
};
static const char *const ll054[7] = {
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
};
static const char *const ll055[3] = {
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe5\x87\xba\xe5\xb8\x88",
};
static const char *const ll056[2] = {
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe4\xb8\x8a\xe5\x86\x8c",
};
static const char *const ll057[25] = {
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\x95\xb4\xe5\xae\xb9",
    "\xe5\x89\x83\xe5\xa4\xb4",
    "\xe6\x95\xb4\xe6\x89\x8b\xe8\xb6\xb3\xe7\x94\xb2",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe7\xa0\xb4\xe5\xb1\x8b\xe5\x9d\x8f\xe5\x9e\xa3",
    "\xe4\xbc\x90\xe6\x9c\xa8",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\xa0\xb4\xe5\x9c\x9f",
    "\xe5\xae\x89\xe8\x91\xac",
    "\xe5\x90\xaf\xe6\x94\x92",
};
static const char *const ll058[1] = {
    "\xe7\xa0\xb4\xe5\xb1\x8b\xe5\x9d\x8f\xe5\x9e\xa3",
};
static const char *const ll059[57] = {
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xb8\x83\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x86\xa0\xe5\xb8\xa6",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xae\x89\xe5\xba\x8a",
    "\xe6\x95\xb4\xe5\xae\xb9",
    "\xe5\x89\x83\xe5\xa4\xb4",
    "\xe6\x95\xb4\xe6\x89\x8b\xe8\xb6\xb3\xe7\x94\xb2",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe7\xad\x91\xe5\xa0\xa4\xe9\x98\xb2",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe9\xbc\x93\xe9\x93\xb8",
    "\xe7\xbb\x8f\xe7\xbb\x9c",
    "\xe9\x85\x9d\xe9\x85\xbf",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe5\xbc\x80\xe6\xb8\xa0",
    "\xe7\xa9\xbf\xe4\xba\x95",
    "\xe5\xae\x89\xe7\xa2\x93\xe7\xa1\x99",
    "\xe5\xa1\x9e\xe7\xa9\xb4",
    "\xe8\xa1\xa5\xe5\x9e\xa3",
    "\xe4\xbf\xae\xe9\xa5\xb0\xe5\x9e\xa3\xe5\xa2\x99",
    "\xe4\xbc\x90\xe6\x9c\xa8",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
    "\xe7\xa0\xb4\xe5\x9c\x9f",
    "\xe5\xae\x89\xe8\x91\xac",
    "\xe5\x90\xaf\xe6\x94\x92",
};
static const char *const ll060[57] = {
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xb8\x83\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x86\xa0\xe5\xb8\xa6",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xae\x89\xe5\xba\x8a",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\x95\xb4\xe5\xae\xb9",
    "\xe5\x89\x83\xe5\xa4\xb4",
    "\xe6\x95\xb4\xe6\x89\x8b\xe8\xb6\xb3\xe7\x94\xb2",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe7\xad\x91\xe5\xa0\xa4\xe9\x98\xb2",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe9\xbc\x93\xe9\x93\xb8",
    "\xe7\xbb\x8f\xe7\xbb\x9c",
    "\xe9\x85\x9d\xe9\x85\xbf",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe5\xbc\x80\xe6\xb8\xa0",
    "\xe7\xa9\xbf\xe4\xba\x95",
    "\xe5\xae\x89\xe7\xa2\x93\xe7\xa1\x99",
    "\xe5\xa1\x9e\xe7\xa9\xb4",
    "\xe8\xa1\xa5\xe5\x9e\xa3",
    "\xe4\xbf\xae\xe9\xa5\xb0\xe5\x9e\xa3\xe5\xa2\x99",
    "\xe7\xa0\xb4\xe5\xb1\x8b\xe5\x9d\x8f\xe5\x9e\xa3",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
    "\xe5\xae\x89\xe8\x91\xac",
};
static const char *const ll061[27] = {
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe7\xbb\x8f\xe7\xbb\x9c",
    "\xe9\x85\x9d\xe9\x85\xbf",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
    "\xe7\xa0\xb4\xe5\x9c\x9f",
    "\xe5\xae\x89\xe8\x91\xac",
    "\xe5\x90\xaf\xe6\x94\x92",
};
static const char *const ll062[59] = {
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xb8\x83\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x86\xa0\xe5\xb8\xa6",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xae\x89\xe5\xba\x8a",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\x95\xb4\xe5\xae\xb9",
    "\xe5\x89\x83\xe5\xa4\xb4",
    "\xe6\x95\xb4\xe6\x89\x8b\xe8\xb6\xb3\xe7\x94\xb2",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe7\xad\x91\xe5\xa0\xa4\xe9\x98\xb2",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe9\xbc\x93\xe9\x93\xb8",
    "\xe7\xbb\x8f\xe7\xbb\x9c",
    "\xe9\x85\x9d\xe9\x85\xbf",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe5\xbc\x80\xe6\xb8\xa0",
    "\xe7\xa9\xbf\xe4\xba\x95",
    "\xe5\xae\x89\xe7\xa2\x93\xe7\xa1\x99",
    "\xe5\xa1\x9e\xe7\xa9\xb4",
    "\xe8\xa1\xa5\xe5\x9e\xa3",
    "\xe4\xbf\xae\xe9\xa5\xb0\xe5\x9e\xa3\xe5\xa2\x99",
    "\xe7\xa0\xb4\xe5\xb1\x8b\xe5\x9d\x8f\xe5\x9e\xa3",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
    "\xe7\xa0\xb4\xe5\x9c\x9f",
    "\xe5\xae\x89\xe8\x91\xac",
    "\xe5\x90\xaf\xe6\x94\x92",
};
static const char *const ll063[62] = {
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xb8\x83\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x86\xa0\xe5\xb8\xa6",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe8\xbf\x9c\xe5\x9b\x9e",
    "\xe5\xae\x89\xe5\xba\x8a",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\x95\xb4\xe5\xae\xb9",
    "\xe5\x89\x83\xe5\xa4\xb4",
    "\xe6\x95\xb4\xe6\x89\x8b\xe8\xb6\xb3\xe7\x94\xb2",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe7\xad\x91\xe5\xa0\xa4\xe9\x98\xb2",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe9\xbc\x93\xe9\x93\xb8",
    "\xe7\xbb\x8f\xe7\xbb\x9c",
    "\xe9\x85\x9d\xe9\x85\xbf",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe5\xbc\x80\xe6\xb8\xa0",
    "\xe7\xa9\xbf\xe4\xba\x95",
    "\xe5\xae\x89\xe7\xa2\x93\xe7\xa1\x99",
    "\xe5\xa1\x9e\xe7\xa9\xb4",
    "\xe8\xa1\xa5\xe5\x9e\xa3",
    "\xe4\xbf\xae\xe9\xa5\xb0\xe5\x9e\xa3\xe5\xa2\x99",
    "\xe5\xb9\xb3\xe6\xb2\xbb\xe9\x81\x93\xe6\xb6\x82",
    "\xe7\xa0\xb4\xe5\xb1\x8b\xe5\x9d\x8f\xe5\x9e\xa3",
    "\xe4\xbc\x90\xe6\x9c\xa8",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
    "\xe7\xa0\xb4\xe5\x9c\x9f",
    "\xe5\xae\x89\xe8\x91\xac",
    "\xe5\x90\xaf\xe6\x94\x92",
};
static const char *const ll064[2] = {
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe4\xb9\x98\xe8\x88\xb9\xe6\xb8\xa1\xe6\xb0\xb4",
};
static const char *const ll065[3] = {
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
};
static const char *const ll066[2] = {
    "\xe5\x8f\x96\xe9\xb1\xbc",
    "\xe4\xb9\x98\xe8\x88\xb9\xe6\xb8\xa1\xe6\xb0\xb4",
};
static const char *const ll067[18] = {
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe6\x8d\x95\xe6\x8d\x89",
    "\xe7\x95\x8b\xe7\x8c\x8e",
    "\xe5\x8f\x96\xe9\xb1\xbc",
};
static const char *const ll068[2] = {
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe5\xae\x89\xe8\x91\xac",
};
static const char *const ll069[4] = {
    "\xe5\xbc\x80\xe5\xbc\xa0",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe7\xab\x8b\xe5\x88\xb8",
};
static const char *const ll070[2] = {
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe5\x85\xa5\xe5\xae\x85",
};
static const char *const ll071[5] = {
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
};
static const char *const ll072[37] = {
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe5\x86\xa0\xe5\xb8\xa6",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xae\x89\xe5\xba\x8a",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe7\xad\x91\xe5\xa0\xa4\xe9\x98\xb2",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
};
static const char *const ll073[3] = {
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe7\xba\xb3\xe7\x95\x9c",
};
static const char *const ll074[1] = {
    "\xe8\x8b\xab\xe7\x9b\x96",
};
static const char *const ll075[1] = {
    "\xe7\xa5\xad\xe7\xa5\x80",
};
static const char *const ll076[3] = {
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe5\xbc\x80\xe4\xbb\x93",
};
static const char *const ll077[16] = {
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe7\xad\x91\xe5\xa0\xa4\xe9\x98\xb2",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe5\xbc\x80\xe6\xb8\xa0",
    "\xe7\xa9\xbf\xe4\xba\x95",
    "\xe5\xae\x89\xe7\xa2\x93\xe7\xa1\x99",
    "\xe8\xa1\xa5\xe5\x9e\xa3",
    "\xe4\xbf\xae\xe9\xa5\xb0\xe5\x9e\xa3\xe5\xa2\x99",
    "\xe5\xb9\xb3\xe6\xb2\xbb\xe9\x81\x93\xe6\xb6\x82",
    "\xe7\xa0\xb4\xe5\xb1\x8b\xe5\x9d\x8f\xe5\x9e\xa3",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\xa0\xb4\xe5\x9c\x9f",
};
static const char *const ll078[12] = {
    "\xe8\xbe\x9b\xe6\x9c\xaa\xe8\xbe\x9b\xe9\x85\x89",
    "\xe4\xb9\x99\xe9\x85\x89\xe4\xb9\x99\xe6\x9c\xaa",
    "\xe5\xba\x9a\xe5\xad\x90\xe5\xba\x9a\xe5\x8d\x88",
    "\xe7\x99\xb8\xe6\x9c\xaa\xe7\x99\xb8\xe4\xb8\x91",
    "\xe7\x94\xb2\xe5\xad\x90\xe7\x94\xb2\xe5\xaf\x85",
    "\xe5\xb7\xb1\xe5\x8d\xaf\xe5\xb7\xb1\xe4\xb8\x91",
    "\xe6\x88\x8a\xe8\xbe\xb0\xe6\x88\x8a\xe5\x8d\x88",
    "\xe7\x99\xb8\xe6\x9c\xaa\xe7\x99\xb8\xe5\xb7\xb3",
    "\xe4\xb8\x99\xe5\xaf\x85\xe4\xb8\x99\xe7\x94\xb3",
    "\xe4\xb8\x81\xe5\x8d\xaf\xe4\xb8\x81\xe5\xb7\xb3",
    "\xe6\x88\x8a\xe8\xbe\xb0\xe6\x88\x8a\xe5\xad\x90",
    "\xe5\xba\x9a\xe6\x88\x8c\xe5\xba\x9a\xe5\xad\x90",
};
static const char *const ll079[1] = {
    "\xe6\xa0\xbd\xe7\xa7\x8d",
};
static const char *const ll080[1] = {
    "\xe7\x95\x8b\xe7\x8c\x8e",
};
static const char *const ll081[1] = {
    "\xe5\x87\xba\xe8\xa1\x8c",
};
static const char *const ll082[4] = {
    "\xe7\xba\xb3\xe7\x95\x9c",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xab\x81\xe5\xa8\xb6",
};
static const char *const ll083[6] = {
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
};
static const char *const ll084[4] = {
    "\xe4\xb8\x81\xe4\xb8\x91\xe5\xb7\xb1\xe9\x85\x89",
    "\xe7\x94\xb2\xe7\x94\xb3\xe7\x94\xb2\xe8\xbe\xb0",
    "\xe8\xbe\x9b\xe6\x9c\xaa\xe4\xb8\x81\xe6\x9c\xaa",
    "\xe7\x94\xb2\xe6\x88\x8c\xe7\x94\xb2\xe5\xaf\x85",
};
static const char *const ll085[1] = {
    "\xe9\x92\x88\xe5\x88\xba",
};
static const char *const ll086[2] = {
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
};
static const char *const ll087[8] = {
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xa5\xad\xe7\xa5\x80",
};
static const char *const ll088[2] = {
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xab\x81\xe5\xa8\xb6",
};
static const char *const ll089[8] = {
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
};
static const char *const ll090[4] = {
    "\xe5\xa3\xac\xe5\xad\x90",
    "\xe4\xb9\x99\xe5\x8d\xaf",
    "\xe6\x88\x8a\xe5\x8d\x88",
    "\xe8\xbe\x9b\xe9\x85\x89",
};
static const char *const ll091[13] = {
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe5\xae\x89\xe8\x91\xac",
};
static const char *const ll092[4] = {
    "\xe4\xb9\x99\xe4\xba\xa5",
    "\xe4\xb8\x81\xe4\xba\xa5",
    "\xe8\xbe\x9b\xe4\xba\xa5",
    "\xe7\x99\xb8\xe4\xba\xa5",
};
static const char *const ll093[7] = {
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe5\xae\x89\xe8\x91\xac",
};
static const char *const ll094[4] = {
    "\xe7\x94\xb2\xe5\xad\x90",
    "\xe4\xb8\x99\xe5\xad\x90",
    "\xe5\xba\x9a\xe5\xad\x90",
    "\xe5\xa3\xac\xe5\xad\x90",
};
static const char *const ll095[55] = {
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xb8\x83\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x86\xa0\xe5\xb8\xa6",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xae\x89\xe5\xba\x8a",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe7\xad\x91\xe5\xa0\xa4\xe9\x98\xb2",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe9\xbc\x93\xe9\x93\xb8",
    "\xe7\xbb\x8f\xe7\xbb\x9c",
    "\xe9\x85\x9d\xe9\x85\xbf",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe5\xbc\x80\xe6\xb8\xa0",
    "\xe7\xa9\xbf\xe4\xba\x95",
    "\xe5\xae\x89\xe7\xa2\x93\xe7\xa1\x99",
    "\xe5\xa1\x9e\xe7\xa9\xb4",
    "\xe8\xa1\xa5\xe5\x9e\xa3",
    "\xe4\xbf\xae\xe9\xa5\xb0\xe5\x9e\xa3\xe5\xa2\x99",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
    "\xe7\xa0\xb4\xe5\x9c\x9f",
    "\xe5\xae\x89\xe8\x91\xac",
    "\xe5\x90\xaf\xe6\x94\x92",
};
static const char *const ll096[4] = {
    "\xe5\xba\x9a\xe7\x94\xb3\xe8\xbe\x9b\xe9\x85\x89",
    "\xe5\xa3\xac\xe5\xad\x90\xe7\x99\xb8\xe4\xba\xa5",
    "\xe7\x94\xb2\xe5\xaf\x85\xe4\xb9\x99\xe5\x8d\xaf",
    "\xe4\xb8\x81\xe5\xb7\xb3\xe4\xb8\x99\xe5\x8d\x88",
};
static const char *const ll097[30] = {
    "\xe5\x86\xa0\xe5\xb8\xa6",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xae\x89\xe5\xba\x8a",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe7\xad\x91\xe5\xa0\xa4\xe9\x98\xb2",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
    "\xe7\xa0\xb4\xe5\x9c\x9f",
    "\xe5\xae\x89\xe8\x91\xac",
    "\xe5\x90\xaf\xe6\x94\x92",
};
static const char *const ll098[12] = {
    "\xe5\xa3\xac\xe8\xbe\xb0",
    "\xe6\x88\x8a\xe8\xbe\xb0",
    "\xe4\xb9\x99\xe6\x9c\xaa",
    "\xe4\xb9\x99\xe6\x9c\xaa",
    "\xe6\x88\x8a\xe8\xbe\xb0",
    "\xe4\xb8\x99\xe6\x88\x8c",
    "\xe4\xb8\x99\xe6\x88\x8c",
    "\xe6\x88\x8a\xe8\xbe\xb0",
    "\xe8\xbe\x9b\xe4\xb8\x91",
    "\xe8\xbe\x9b\xe4\xb8\x91",
    "\xe6\x88\x8a\xe8\xbe\xb0",
    "\xe5\xa3\xac\xe8\xbe\xb0",
};
static const char *const ll099[2] = {
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe5\xbc\x80\xe4\xbb\x93",
};
static const char *const ll100[4] = {
    "\xe5\xb7\xb3\xe9\x85\x89\xe4\xb8\x91",
    "\xe7\x94\xb3\xe5\xad\x90\xe8\xbe\xb0",
    "\xe4\xba\xa5\xe5\x8d\xaf\xe6\x9c\xaa",
    "\xe5\xaf\x85\xe5\x8d\x88\xe6\x88\x8c",
};
static const char *const ll101[1] = {
    "\xe6\xb2\x90\xe6\xb5\xb4",
};
static const char *const ll102[5] = {
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
};
static const char *const ll103[6] = {
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
};
static const char *const ll104[5] = {
    "\xe4\xb8\x81\xe6\x9c\xaa",
    "\xe5\xb7\xb1\xe6\x9c\xaa",
    "\xe5\xba\x9a\xe7\x94\xb3",
    "\xe7\x94\xb2\xe5\xaf\x85",
    "\xe7\x99\xb8\xe4\xb8\x91",
};
static const char *const ll105[4] = {
    "\xe5\xa1\x9e\xe7\xa9\xb4",
    "\xe8\xa1\xa5\xe5\x9e\xa3",
    "\xe5\x8f\x96\xe9\xb1\xbc",
    "\xe4\xb9\x98\xe8\x88\xb9\xe6\xb8\xa1\xe6\xb0\xb4",
};
static const char *const ll106[2] = {
    "\xe9\xbc\x93\xe9\x93\xb8",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
};
static const char *const ll107[3] = {
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xab\x81\xe5\xa8\xb6",
};
static const char *const ll108[3] = {
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe5\x8f\x96\xe9\xb1\xbc",
    "\xe4\xb9\x98\xe8\x88\xb9\xe6\xb8\xa1\xe6\xb0\xb4",
};
static const char *const ll109[4] = {
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
};
static const char *const ll110[2] = {
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe8\xbf\x9c\xe5\x9b\x9e",
};
static const char *const ll111[4] = {
    "\xe6\x88\x8a\xe5\x8d\x88",
    "\xe4\xb8\x99\xe5\x8d\x88",
    "\xe5\xa3\xac\xe5\xad\x90",
    "\xe6\x88\x8a\xe5\xad\x90",
};
static const char *const ll112[1] = {
    "\xe4\xb8\x81\xe5\xb7\xb3",
};
static const char *const ll113[12] = {
    "\xe4\xb8\x81\xe6\x9c\xaa",
    "\xe4\xb8\x81\xe4\xb8\x91",
    "\xe4\xb8\x99\xe6\x88\x8c",
    "\xe7\x94\xb2\xe5\x8d\x88",
    "\xe5\xba\x9a\xe5\xad\x90",
    "\xe5\xa3\xac\xe5\xaf\x85",
    "\xe7\x99\xb8\xe5\x8d\xaf",
    "\xe4\xb9\x99\xe5\xb7\xb3",
    "\xe6\x88\x8a\xe7\x94\xb3",
    "\xe5\xb7\xb1\xe9\x85\x89",
    "\xe8\xbe\x9b\xe4\xba\xa5",
    "\xe4\xb8\x99\xe8\xbe\xb0",
};
static const char *const ll114[12] = {
    "\xe7\x94\xb2\xe5\xad\x90",
    "\xe4\xb8\x99\xe5\xaf\x85",
    "\xe4\xb8\x81\xe5\x8d\xaf",
    "\xe5\xb7\xb1\xe5\xb7\xb3",
    "\xe8\xbe\x9b\xe6\x9c\xaa",
    "\xe5\xa3\xac\xe7\x94\xb3",
    "\xe7\x99\xb8\xe9\x85\x89",
    "\xe4\xb9\x99\xe4\xba\xa5",
    "\xe5\xba\x9a\xe8\xbe\xb0",
    "\xe8\xbe\x9b\xe4\xb8\x91",
    "\xe5\xba\x9a\xe6\x88\x8c",
    "\xe6\x88\x8a\xe5\x8d\x88",
};
static const char *const ll115[12] = {
    "\xe4\xb9\x99\xe4\xb8\x91",
    "\xe7\x94\xb2\xe6\x88\x8c",
    "\xe5\xa3\xac\xe5\x8d\x88",
    "\xe6\x88\x8a\xe5\xad\x90",
    "\xe5\xba\x9a\xe5\xaf\x85",
    "\xe8\xbe\x9b\xe5\x8d\xaf",
    "\xe7\x99\xb8\xe5\xb7\xb3",
    "\xe4\xb9\x99\xe6\x9c\xaa",
    "\xe4\xb8\x99\xe7\x94\xb3",
    "\xe4\xb8\x81\xe9\x85\x89",
    "\xe5\xb7\xb1\xe4\xba\xa5",
    "\xe7\x94\xb2\xe8\xbe\xb0",
};
static const char *const ll116[12] = {
    "\xe5\xba\x9a\xe5\x8d\x88",
    "\xe8\xbe\x9b\xe5\xb7\xb3",
    "\xe4\xb8\x99\xe5\xad\x90",
    "\xe6\x88\x8a\xe5\xaf\x85",
    "\xe5\xb7\xb1\xe5\x8d\xaf",
    "\xe7\x99\xb8\xe6\x9c\xaa",
    "\xe7\x99\xb8\xe4\xb8\x91",
    "\xe7\x94\xb2\xe7\x94\xb3",
    "\xe4\xb9\x99\xe9\x85\x89",
    "\xe4\xb8\x81\xe4\xba\xa5",
    "\xe5\xa3\xac\xe8\xbe\xb0",
    "\xe5\xa3\xac\xe6\x88\x8c",
};
static const char *const ll117[12] = {
    "\xe7\x94\xb2\xe5\xaf\x85",
    "\xe4\xb9\x99\xe5\x8d\xaf",
    "\xe4\xb8\x81\xe5\xb7\xb3",
    "\xe4\xb8\x99\xe5\x8d\x88",
    "\xe5\xba\x9a\xe7\x94\xb3",
    "\xe8\xbe\x9b\xe9\x85\x89",
    "\xe7\x99\xb8\xe4\xba\xa5",
    "\xe5\xa3\xac\xe5\xad\x90",
    "\xe6\x88\x8a\xe8\xbe\xb0",
    "\xe6\x88\x8a\xe6\x88\x8c",
    "\xe5\xb7\xb1\xe4\xb8\x91",
    "\xe5\xb7\xb1\xe6\x9c\xaa",
};
static const char *const ll118[3] = {
    "\xe7\xa0\xb4\xe5\x9c\x9f",
    "\xe5\xae\x89\xe8\x91\xac",
    "\xe5\x90\xaf\xe6\x94\x92",
};
static const char *const ll119[1] = {
    "\xe8\xa3\x81\xe5\x88\xb6",
};
static const char *const ll120[6] = {
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
};
static const char *const ll121[10] = {
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe7\xbb\x8f\xe7\xbb\x9c",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe5\xa1\x9e\xe7\xa9\xb4",
    "\xe8\xa1\xa5\xe5\x9e\xa3",
};
static const char *const ll122[8] = {
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
};
static const char *const ll123[2] = {
    "\xe4\xbf\xae\xe9\xa5\xb0\xe5\x9e\xa3\xe5\xa2\x99",
    "\xe5\xb9\xb3\xe6\xb2\xbb\xe9\x81\x93\xe6\xb6\x82",
};
static const char *const ll124[51] = {
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xb8\x83\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x86\xa0\xe5\xb8\xa6",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xae\x89\xe5\xba\x8a",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe7\xad\x91\xe5\xa0\xa4\xe9\x98\xb2",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe9\xbc\x93\xe9\x93\xb8",
    "\xe7\xbb\x8f\xe7\xbb\x9c",
    "\xe9\x85\x9d\xe9\x85\xbf",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe5\xbc\x80\xe6\xb8\xa0",
    "\xe7\xa9\xbf\xe4\xba\x95",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
    "\xe7\xba\xb3\xe7\x95\x9c",
    "\xe7\xa0\xb4\xe5\x9c\x9f",
    "\xe5\xae\x89\xe8\x91\xac",
    "\xe5\x90\xaf\xe6\x94\x92",
};
static const char *const ll125[1] = {
    "\xe5\x86\xa0\xe5\xb8\xa6",
};
static const char *const ll126[1] = {
    "\xe6\x8d\x95\xe6\x8d\x89",
};
static const char *const ll127[3] = {
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\xae\x89\xe5\xba\x8a",
};
static const char *const ll128[5] = {
    "\xe5\x85\xa5\xe5\xad\xa6",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe7\xad\x91\xe5\xa0\xa4\xe9\x98\xb2",
    "\xe5\xbc\x80\xe5\xb8\x82",
};
static const char *const ll129[4] = {
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe7\xba\xb3\xe8\xb4\xa2",
    "\xe6\x8d\x95\xe6\x8d\x89",
    "\xe7\xba\xb3\xe7\x95\x9c",
};
static const char *const ll130[45] = {
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xb8\x83\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x86\xa0\xe5\xb8\xa6",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\xae\x89\xe6\x8a\x9a\xe8\xbe\xb9\xe5\xa2\x83",
    "\xe9\x80\x89\xe5\xb0\x86",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xae\x89\xe5\xba\x8a",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe7\xad\x91\xe5\xa0\xa4\xe9\x98\xb2",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe9\xbc\x93\xe9\x93\xb8",
    "\xe7\xbb\x8f\xe7\xbb\x9c",
    "\xe9\x85\x9d\xe9\x85\xbf",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe5\xbc\x80\xe6\xb8\xa0",
    "\xe7\xa9\xbf\xe4\xba\x95",
    "\xe7\xa0\xb4\xe5\x9c\x9f",
    "\xe5\xae\x89\xe8\x91\xac",
    "\xe5\x90\xaf\xe6\x94\x92",
};
static const char *const ll131[34] = {
    "\xe7\xa5\xad\xe7\xa5\x80",
    "\xe7\xa5\x88\xe7\xa6\x8f",
    "\xe6\xb1\x82\xe5\x97\xa3",
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe8\xa6\x83\xe6\x81\xa9",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe6\x81\xa4\xe5\xad\xa4\xe8\x8c\x95",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe9\x9b\xaa\xe5\x86\xa4",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x85\xa5\xe5\xad\xa6",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe8\xa7\xa3\xe9\x99\xa4",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe8\xa3\x81\xe5\x88\xb6",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe5\xbc\x80\xe6\xb8\xa0",
    "\xe7\xa9\xbf\xe4\xba\x95",
    "\xe5\xae\x89\xe7\xa2\x93\xe7\xa1\x99",
    "\xe6\xa0\xbd\xe7\xa7\x8d",
    "\xe7\x89\xa7\xe5\x85\xbb",
};
static const char *const ll132[3] = {
    "\xe7\xad\x91\xe5\xa0\xa4\xe9\x98\xb2",
    "\xe5\xa1\x9e\xe7\xa9\xb4",
    "\xe8\xa1\xa5\xe5\x9e\xa3",
};
static const char *const ll133[31] = {
    "\xe4\xb8\x8a\xe5\x86\x8c",
    "\xe4\xb8\x8a\xe8\xa1\xa8\xe7\xab\xa0",
    "\xe9\xa2\x81\xe8\xaf\x8f",
    "\xe6\x96\xbd\xe6\x81\xa9",
    "\xe6\x8b\x9b\xe8\xb4\xa4",
    "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4",
    "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xb8\x83\xe6\x94\xbf\xe4\xba\x8b",
    "\xe5\xba\x86\xe8\xb5\x90",
    "\xe5\xae\xb4\xe4\xbc\x9a",
    "\xe5\x87\xba\xe8\xa1\x8c",
    "\xe5\x87\xba\xe5\xb8\x88",
    "\xe4\xb8\x8a\xe5\xae\x98",
    "\xe4\xb8\xb4\xe6\x94\xbf",
    "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb",
    "\xe7\xba\xb3\xe9\x87\x87",
    "\xe5\xab\x81\xe5\xa8\xb6",
    "\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3",
    "\xe6\x90\xac\xe7\xa7\xbb",
    "\xe5\xae\x89\xe5\xba\x8a",
    "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85",
    "\xe7\x96\x97\xe7\x9b\xae",
    "\xe8\x90\xa5\xe5\xbb\xba",
    "\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4",
    "\xe4\xbf\xae\xe9\x80\xa0",
    "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81",
    "\xe5\xbc\x80\xe5\xb8\x82",
    "\xe5\xbc\x80\xe4\xbb\x93",
    "\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4",
    "\xe5\xbc\x80\xe6\xb8\xa0",
    "\xe7\xa9\xbf\xe4\xba\x95",
};
static const char *const ll134[1] = {
    "\xe5\xbc\x80\xe4\xbb\x93",
};
static const char *const ll135[2] = {
    "\xe6\x95\xb4\xe5\xae\xb9",
    "\xe5\x89\x83\xe5\xa4\xb4",
};
static const char *const ll136[1] = {
    "\xe7\xbb\x8f\xe7\xbb\x9c",
};
static const char *const ll137[1] = {
    "\xe9\x85\x9d\xe9\x85\xbf",
};
static const char *const ll138[2] = {
    "\xe5\xbc\x80\xe6\xb8\xa0",
    "\xe7\xa9\xbf\xe4\xba\x95",
};
static const char *const ll139[1] = {
    "\xe7\xa9\xbf\xe4\xba\x95",
};
static const char *const ll140[1] = {
    "\xe5\xae\xb4\xe4\xbc\x9a",
};
static const char *const ll141[1] = {
    "\xe5\xae\x89\xe5\xba\x8a",
};
static const char *const ll142[3] = {
    "\xe7\x9b\xb8\xe6\x97\xa5",
    "\xe6\x97\xb6\xe5\xbe\xb7",
    "\xe5\x85\xad\xe5\x90\x88",
};
static const char *const ll143[3] = {
    "\xe7\x9b\xb8\xe6\x97\xa5",
    "\xe5\x85\xad\xe5\x90\x88",
    "\xe6\x9c\x88\xe5\x88\x91",
};
static const char *const ll144[2] = {
    "\xe7\x9b\xb8\xe6\x97\xa5",
    "\xe6\x9c\x88\xe5\xae\xb3",
};
static const char *const ll145[3] = {
    "\xe7\x9b\xb8\xe6\x97\xa5",
    "\xe6\x9c\x88\xe5\xae\xb3",
    "\xe6\x9c\x88\xe5\x88\x91",
};
static const char *const ll146[1] = {
    "\xe5\xa4\xa9\xe5\x90\x8f",
};
static const char *const ll147[1] = {
    "\xe6\x9c\x88\xe7\x85\x9e",
};
static const char *const ll148[2] = {
    "\xe5\xa4\xa9\xe5\x90\x8f",
    "\xe6\x9c\x88\xe5\x88\x91",
};
static const char *const ll149[3] = {
    "\xe9\x95\xbf\xe7\x94\x9f",
    "\xe5\x85\xad\xe5\x90\x88",
    "\xe5\x8a\xab\xe7\x85\x9e",
};
static const char *const ll150[2] = {
    "\xe9\x95\xbf\xe7\x94\x9f",
    "\xe5\x8a\xab\xe7\x85\x9e",
};
static const char *const ll151[1] = {
    "\xe6\x9c\x88\xe5\xae\xb3",
};
static const char *const ll152[1] = {
    "\xe5\xa4\xa7\xe6\x97\xb6",
};
static const char *const ll153[1] = {
    "\xe6\x9c\x88\xe5\x88\x91",
};
static const char *const ll154[1] = {
    "\xe7\x8e\x8b\xe6\x97\xa5",
};
static const char *const ll155[2] = {
    "\xe5\xae\x98\xe6\x97\xa5",
    "\xe5\xa4\xa9\xe5\x90\x8f",
};
static const char *const ll156[2] = {
    "\xe9\x95\xbf\xe7\x94\x9f",
    "\xe5\x85\xad\xe5\x90\x88",
};
static const char *const ll157[2] = {
    "\xe9\x99\xa4\xe6\x97\xa5",
    "\xe7\x9b\xb8\xe6\x97\xa5",
};
static const char *const ll158[2] = {
    "\xe9\x95\xbf\xe7\x94\x9f",
    "\xe6\x9c\x88\xe5\xae\xb3",
};
static const char *const ll159[1] = {
    "\xe6\x89\xa7\xe6\x97\xa5",
};
static const char *const ll160[1] = {
    "\xe5\xbc\x80\xe6\x97\xa5",
};
static const char *const ll161[2] = {
    "\xe6\xbb\xa1\xe6\x97\xa5",
    "\xe6\xb0\x91\xe6\x97\xa5",
};
static const char *const ll162[1] = {
    "\xe6\x9c\x88\xe7\xa0\xb4",
};
static const char *const ll163[2] = {
    "\xe6\x9c\x88\xe7\xa0\xb4",
    "\xe6\x9c\x88\xe5\x8e\x8c",
};
static const char *const ll164[2] = {
    "\xe5\x85\xad\xe5\x90\x88",
    "\xe5\x8d\xb1\xe6\x97\xa5",
};
static const char *const ll165[2] = {
    "\xe6\x9c\x88\xe5\xae\xb3",
    "\xe5\x8d\xb1\xe6\x97\xa5",
};
static const char *const ll166[3] = {
    "\xe5\xb9\xb3\xe6\x97\xa5",
    "\xe5\x85\xad\xe5\x90\x88",
    "\xe7\x9b\xb8\xe6\x97\xa5",
};
static const char *const ll167[3] = {
    "\xe7\x9b\xb8\xe6\x97\xa5",
    "\xe6\x9c\x88\xe5\xae\xb3",
    "\xe5\xb9\xb3\xe6\x97\xa5",
};
static const char *const ll168[1] = {
    "\xe5\xbb\xba\xe6\x97\xa5",
};
static const char *const ll169[2] = {
    "\xe5\xb9\xb3\xe6\x97\xa5",
    "\xe5\xa4\xa9\xe5\x90\x8f",
};
static const char *const ll170[3] = {
    "\xe6\x94\xb6\xe6\x97\xa5",
    "\xe5\xa4\xa7\xe6\x97\xb6",
    "\xe5\xa4\xa9\xe7\xa0\xb4",
};
static const char *const ll171[3] = {
    "\xe6\x9c\x88\xe5\xbb\xba",
    "\xe6\x9c\x88\xe5\x8e\x8c",
    "\xe5\xbe\xb7\xe5\xa4\xa7\xe4\xbc\x9a",
};
static const char *const ll172[2] = {
    "\xe5\xae\x88\xe6\x97\xa5",
    "\xe9\x99\xa4\xe6\x97\xa5",
};
static const char *const ll173[2] = {
    "\xe6\x89\xa7\xe6\x97\xa5",
    "\xe5\xa4\xa7\xe6\x97\xb6",
};
static const char *const ll174[2] = {
    "\xe7\x9b\xb8\xe6\x97\xa5",
    "\xe5\xb9\xb3\xe6\x97\xa5",
};
static const char *const ll175[3] = {
    "\xe5\xae\x98\xe6\x97\xa5",
    "\xe9\x97\xad\xe6\x97\xa5",
    "\xe5\xa4\xa9\xe5\x90\x8f",
};
static const char *const ll176[3] = {
    "\xe7\x9b\xb8\xe6\x97\xa5",
    "\xe5\xb9\xb3\xe6\x97\xa5",
    "\xe6\x9c\x88\xe5\x88\x91",
};
static const char *const ll177[1] = {
    "\xe6\x88\x90\xe6\x97\xa5",
};
static const char *const ll178[1] = {
    "\xe5\xae\x9a\xe6\x97\xa5",
};
static const char *const ll179[1] = {
    "\xe6\xbb\xa1\xe6\x97\xa5",
};
static const char *const ll180[2] = {
    "\xe6\x9c\x88\xe5\xbb\xba",
    "\xe5\xbe\xb7\xe5\xa4\xa7\xe4\xbc\x9a",
};
static const char *const ll181[3] = {
    "\xe6\x9c\x88\xe5\xbb\xba",
    "\xe6\x9c\x88\xe5\x88\x91",
    "\xe5\xbe\xb7\xe5\xa4\xa7\xe4\xbc\x9a",
};
static const char *const ll182[2] = {
    "\xe6\x9c\x88\xe7\xa0\xb4",
    "\xe7\x81\xbe\xe7\x85\x9e",
};
static const char *const ll183[2] = {
    "\xe9\x99\xa4\xe6\x97\xa5",
    "\xe5\xae\x98\xe6\x97\xa5",
};
static const char *const ll184[2] = {
    "\xe6\x89\xa7\xe6\x97\xa5",
    "\xe5\x85\xad\xe5\x90\x88",
};
static const char *const ll185[2] = {
    "\xe6\x89\xa7\xe6\x97\xa5",
    "\xe6\x9c\x88\xe5\xae\xb3",
};
static const char *const ll186[1] = {
    "\xe6\x94\xb6\xe6\x97\xa5",
};
static const char *const ll187[2] = {
    "\xe6\x94\xb6\xe6\x97\xa5",
    "\xe6\x9c\x88\xe5\x88\x91",
};
static const char *const ll188[1] = {
    "\xe5\x8d\xb1\xe6\x97\xa5",
};
static const char *const ll189[1] = {
    "\xe9\x97\xad\xe6\x97\xa5",
};
static const char *const ll190[1] = {
    "\xe5\xb9\xb3\xe6\x97\xa5",
};
static const char *const ll191[2] = {
    "\xe5\xb9\xb3\xe6\x97\xa5",
    "\xe6\x9c\x88\xe5\x88\x91",
};
static const int pi000[3] = {
    5,
    14,
    23,
};
static const int pi001[26] = {
    1,
    13,
    2,
    11,
    3,
    9,
    4,
    7,
    5,
    5,
    6,
    2,
    7,
    1,
    7,
    29,
    8,
    27,
    9,
    25,
    10,
    23,
    11,
    21,
    12,
    19,
};
static const int pi002[6] = {
    3,
    7,
    13,
    18,
    22,
    27,
};
static const int pi003[4] = {
    4,
    4,
    10,
    10,
};
static const int pi004[4] = {
    6,
    6,
    12,
    12,
};
static const int pi005[1] = {
    10,
};

static const cnl_god_row cnl_ANGEL_ROWS[71] = {
    { "岁德", 1, "\xe7\x94\xb2\xe5\xba\x9a\xe4\xb8\x99\xe5\xa3\xac\xe6\x88\x8a\xe7\x94\xb2\xe5\xba\x9a\xe4\xb8\x99\xe5\xa3\xac\xe6\x88\x8a", 2, NULL, 0, NULL, NULL, 0, ll000, cnl_noarr, 1, 0 },
    { "岁德合", 1, "\xe5\xb7\xb1\xe4\xb9\x99\xe8\xbe\x9b\xe4\xb8\x81\xe7\x99\xb8\xe5\xb7\xb1\xe4\xb9\x99\xe8\xbe\x9b\xe4\xb8\x81\xe7\x99\xb8", 2, NULL, 0, NULL, NULL, 0, ll000, cnl_noarr, 1, 0 },
    { "月德", 2, "\xe5\xa3\xac\xe5\xba\x9a\xe4\xb8\x99\xe7\x94\xb2\xe5\xa3\xac\xe5\xba\x9a\xe4\xb8\x99\xe7\x94\xb2\xe5\xa3\xac\xe5\xba\x9a\xe4\xb8\x99\xe7\x94\xb2", 0, NULL, 0, NULL, NULL, 0, ll001, ll002, 37, 2 },
    { "月德合", 2, "\xe4\xb8\x81\xe4\xb9\x99\xe8\xbe\x9b\xe5\xb7\xb1\xe4\xb8\x81\xe4\xb9\x99\xe8\xbe\x9b\xe5\xb7\xb1\xe4\xb8\x81\xe4\xb9\x99\xe8\xbe\x9b\xe5\xb7\xb1", 0, NULL, 0, NULL, NULL, 0, ll001, ll002, 37, 2 },
    { "天德", 20, NULL, 0, ll003, 12, NULL, NULL, 0, ll001, ll002, 37, 2 },
    { "天德合", 1, "\xe7\xa9\xba\xe4\xb9\x99\xe5\xa3\xac\xe7\xa9\xba\xe4\xb8\x81\xe4\xb8\x99\xe7\xa9\xba\xe5\xb7\xb1\xe6\x88\x8a\xe7\xa9\xba\xe8\xbe\x9b\xe5\xba\x9a", 0, NULL, 0, NULL, NULL, 0, ll001, ll002, 37, 2 },
    { "凤凰日", 4, "\xe5\x8d\xb1\xe6\x98\xb4\xe8\x83\x83\xe6\xaf\x95", 1, NULL, 0, NULL, NULL, 0, ll004, cnl_noarr, 1, 0 },
    { "麒麟日", 4, "\xe4\xba\x95\xe5\xb0\xbe\xe7\x89\x9b\xe5\xa3\x81", 1, NULL, 0, NULL, NULL, 0, ll004, cnl_noarr, 1, 0 },
    { "三合", 17, NULL, -1, NULL, 0, NULL, NULL, 0, ll005, cnl_noarr, 18, 0 },
    { "四相", 5, NULL, 1, ll007, 4, NULL, NULL, 0, ll006, cnl_noarr, 24, 0 },
    { "五合", 7, "\xe5\xaf\x85\xe5\x8d\xaf", -1, NULL, 0, NULL, NULL, 0, ll008, cnl_noarr, 3, 0 },
    { "五富", 1, "\xe5\xb7\xb3\xe7\x94\xb3\xe4\xba\xa5\xe5\xaf\x85\xe5\xb7\xb3\xe7\x94\xb3\xe4\xba\xa5\xe5\xaf\x85\xe5\xb7\xb3\xe7\x94\xb3\xe4\xba\xa5\xe5\xaf\x85", 0, NULL, 0, NULL, NULL, 0, ll009, cnl_noarr, 9, 0 },
    { "六合", 1, "\xe4\xb8\x91\xe5\xad\x90\xe4\xba\xa5\xe6\x88\x8c\xe9\x85\x89\xe7\x94\xb3\xe6\x9c\xaa\xe5\x8d\x88\xe5\xb7\xb3\xe8\xbe\xb0\xe5\x8d\xaf\xe5\xaf\x85", 0, NULL, 0, NULL, NULL, 0, ll010, cnl_noarr, 10, 0 },
    { "六仪", 1, "\xe5\x8d\x88\xe5\xb7\xb3\xe8\xbe\xb0\xe5\x8d\xaf\xe5\xaf\x85\xe4\xb8\x91\xe5\xad\x90\xe4\xba\xa5\xe6\x88\x8c\xe9\x85\x89\xe7\x94\xb3\xe6\x9c\xaa", 0, NULL, 0, NULL, NULL, 0, ll011, cnl_noarr, 1, 0 },
    { "不将", 21, NULL, 0, NULL, 0, NULL, NULL, 0, ll004, cnl_noarr, 1, 0 },
    { "时德", 3, "\xe5\x8d\x88\xe8\xbe\xb0\xe5\xad\x90\xe5\xaf\x85", 1, NULL, 0, NULL, NULL, 0, ll006, cnl_noarr, 24, 0 },
    { "大葬", 8, NULL, -1, ll013, 12, NULL, NULL, 0, ll012, cnl_noarr, 1, 0 },
    { "鸣吠", 8, NULL, -1, ll015, 14, NULL, NULL, 0, ll014, cnl_noarr, 2, 0 },
    { "小葬", 8, NULL, -1, ll016, 7, NULL, NULL, 0, ll012, cnl_noarr, 1, 0 },
    { "鸣吠对", 8, NULL, -1, ll018, 10, NULL, NULL, 0, ll017, cnl_noarr, 2, 0 },
    { "不守塚", 8, NULL, -1, ll020, 22, NULL, NULL, 0, ll019, cnl_noarr, 1, 0 },
    { "王日", 3, "\xe5\xaf\x85\xe5\xb7\xb3\xe7\x94\xb3\xe4\xba\xa5", 1, NULL, 0, NULL, NULL, 0, ll021, cnl_noarr, 16, 0 },
    { "官日", 3, "\xe5\x8d\xaf\xe5\x8d\x88\xe9\x85\x89\xe5\xad\x90", 1, NULL, 0, NULL, NULL, 0, ll022, cnl_noarr, 2, 0 },
    { "守日", 3, "\xe9\x85\x89\xe5\xad\x90\xe5\x8d\xaf\xe5\x8d\x88", 1, NULL, 0, NULL, NULL, 0, ll023, cnl_noarr, 3, 0 },
    { "相日", 3, "\xe5\xb7\xb3\xe7\x94\xb3\xe4\xba\xa5\xe5\xaf\x85", 1, NULL, 0, NULL, NULL, 0, ll022, cnl_noarr, 2, 0 },
    { "民日", 3, "\xe5\x8d\x88\xe9\x85\x89\xe5\xad\x90\xe5\x8d\xaf", 1, NULL, 0, NULL, NULL, 0, ll024, cnl_noarr, 11, 0 },
    { "临日", 1, "\xe8\xbe\xb0\xe9\x85\x89\xe5\x8d\x88\xe4\xba\xa5\xe7\x94\xb3\xe4\xb8\x91\xe6\x88\x8c\xe5\x8d\xaf\xe5\xad\x90\xe5\xb7\xb3\xe5\xaf\x85\xe6\x9c\xaa", 0, NULL, 0, NULL, NULL, 0, ll025, cnl_noarr, 4, 0 },
    { "天贵", 5, NULL, 1, ll026, 4, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "天喜", 3, "\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa", 0, NULL, 0, NULL, NULL, 0, ll027, cnl_noarr, 10, 0 },
    { "天富", 1, "\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91", 0, NULL, 0, NULL, NULL, 0, ll028, cnl_noarr, 2, 0 },
    { "天恩", 19, NULL, -1, NULL, 0, NULL, NULL, 0, ll029, cnl_noarr, 6, 0 },
    { "月恩", 1, "\xe7\x94\xb2\xe8\xbe\x9b\xe4\xb8\x99\xe4\xb8\x81\xe5\xba\x9a\xe5\xb7\xb1\xe6\x88\x8a\xe8\xbe\x9b\xe5\xa3\xac\xe7\x99\xb8\xe5\xba\x9a\xe4\xb9\x99", 0, NULL, 0, NULL, NULL, 0, ll006, cnl_noarr, 24, 0 },
    { "天赦", 9, NULL, 0, ll031, 12, NULL, NULL, 0, ll030, ll002, 36, 2 },
    { "天愿", 9, NULL, 0, ll033, 12, NULL, NULL, 0, ll032, cnl_noarr, 40, 0 },
    { "天成", 1, "\xe5\x8d\xaf\xe5\xb7\xb3\xe6\x9c\xaa\xe9\x85\x89\xe4\xba\xa5\xe4\xb8\x91\xe5\x8d\xaf\xe5\xb7\xb3\xe6\x9c\xaa\xe9\x85\x89\xe4\xba\xa5\xe4\xb8\x91", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "天官", 1, "\xe5\x8d\x88\xe7\x94\xb3\xe6\x88\x8c\xe5\xad\x90\xe5\xaf\x85\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb3\xe6\x88\x8c\xe5\xad\x90\xe5\xaf\x85\xe8\xbe\xb0", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "天医", 1, "\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c", 0, NULL, 0, NULL, NULL, 0, ll034, cnl_noarr, 1, 0 },
    { "天马", 1, "\xe5\xaf\x85\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb3\xe6\x88\x8c\xe5\xad\x90\xe5\xaf\x85\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb3\xe6\x88\x8c\xe5\xad\x90", 0, NULL, 0, NULL, NULL, 0, ll035, cnl_noarr, 2, 0 },
    { "驿马", 1, "\xe5\xaf\x85\xe4\xba\xa5\xe7\x94\xb3\xe5\xb7\xb3\xe5\xaf\x85\xe4\xba\xa5\xe7\x94\xb3\xe5\xb7\xb3\xe5\xaf\x85\xe4\xba\xa5\xe7\x94\xb3\xe5\xb7\xb3", 0, NULL, 0, NULL, NULL, 0, ll035, cnl_noarr, 2, 0 },
    { "天财", 1, "\xe5\xad\x90\xe5\xaf\x85\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb3\xe6\x88\x8c\xe5\xad\x90\xe5\xaf\x85\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb3\xe6\x88\x8c", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "福生", 1, "\xe5\xaf\x85\xe7\x94\xb3\xe9\x85\x89\xe5\x8d\xaf\xe6\x88\x8c\xe8\xbe\xb0\xe4\xba\xa5\xe5\xb7\xb3\xe5\xad\x90\xe5\x8d\x88\xe4\xb8\x91\xe6\x9c\xaa", 0, NULL, 0, NULL, NULL, 0, ll036, cnl_noarr, 2, 0 },
    { "福厚", 1, "\xe5\xaf\x85\xe5\xb7\xb3\xe7\x94\xb3\xe4\xba\xa5", 1, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "福德", 1, "\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91", 0, NULL, 0, NULL, NULL, 0, ll037, cnl_noarr, 6, 0 },
    { "天巫", 1, "\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91", 0, NULL, 0, NULL, NULL, 0, ll034, cnl_noarr, 1, 0 },
    { "地财", 1, "\xe4\xb8\x91\xe5\x8d\xaf\xe5\xb7\xb3\xe6\x9c\xaa\xe9\x85\x89\xe4\xba\xa5\xe4\xb8\x91\xe5\x8d\xaf\xe5\xb7\xb3\xe6\x9c\xaa\xe9\x85\x89\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "月财", 1, "\xe9\x85\x89\xe4\xba\xa5\xe5\x8d\x88\xe5\xb7\xb3\xe5\xb7\xb3\xe6\x9c\xaa\xe9\x85\x89\xe4\xba\xa5\xe5\x8d\x88\xe5\xb7\xb3\xe5\xb7\xb3\xe6\x9c\xaa", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "月空", 1, "\xe4\xb8\x99\xe7\x94\xb2\xe5\xa3\xac\xe5\xba\x9a\xe4\xb8\x99\xe7\x94\xb2\xe5\xa3\xac\xe5\xba\x9a\xe4\xb8\x99\xe7\x94\xb2\xe5\xa3\xac\xe5\xba\x9a", 0, NULL, 0, NULL, NULL, 0, ll038, cnl_noarr, 1, 0 },
    { "母仓", 6, NULL, 1, ll040, 4, NULL, NULL, 0, ll039, cnl_noarr, 4, 0 },
    { "明星", 1, "\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb2\xe6\x88\x8c\xe5\xad\x90\xe5\xaf\x85\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb2\xe6\x88\x8c\xe5\xad\x90\xe5\xaf\x85", 0, NULL, 0, NULL, NULL, 0, ll041, cnl_noarr, 3, 0 },
    { "圣心", 1, "\xe8\xbe\xb0\xe6\x88\x8c\xe4\xba\xa5\xe5\xb7\xb3\xe5\xad\x90\xe5\x8d\x88\xe4\xb8\x91\xe6\x9c\xaa\xe5\xaf\x85\xe7\x94\xb3\xe5\x8d\xaf\xe9\x85\x89", 0, NULL, 0, NULL, NULL, 0, ll036, cnl_noarr, 2, 0 },
    { "禄库", 1, "\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91", 0, NULL, 0, NULL, NULL, 0, ll042, cnl_noarr, 1, 0 },
    { "吉庆", 1, "\xe6\x9c\xaa\xe5\xad\x90\xe9\x85\x89\xe5\xaf\x85\xe4\xba\xa5\xe8\xbe\xb0\xe4\xb8\x91\xe5\x8d\x88\xe5\x8d\xaf\xe7\x94\xb3\xe5\xb7\xb3\xe6\x88\x8c", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "阴德", 1, "\xe4\xb8\x91\xe4\xba\xa5\xe9\x85\x89\xe6\x9c\xaa\xe5\xb7\xb3\xe5\x8d\xaf\xe4\xb8\x91\xe4\xba\xa5\xe9\x85\x89\xe6\x9c\xaa\xe5\xb7\xb3\xe5\x8d\xaf", 0, NULL, 0, NULL, NULL, 0, ll043, cnl_noarr, 2, 0 },
    { "活曜", 1, "\xe5\x8d\xaf\xe7\x94\xb3\xe5\xb7\xb3\xe6\x88\x8c\xe6\x9c\xaa\xe5\xad\x90\xe9\x85\x89\xe5\xaf\x85\xe4\xba\xa5\xe8\xbe\xb0\xe4\xb8\x91\xe5\x8d\x88", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "除神", 7, "\xe7\x94\xb3\xe9\x85\x89", -1, NULL, 0, NULL, NULL, 0, ll044, cnl_noarr, 7, 0 },
    { "解神", 1, "\xe5\x8d\x88\xe5\x8d\x88\xe7\x94\xb3\xe7\x94\xb3\xe6\x88\x8c\xe6\x88\x8c\xe5\xad\x90\xe5\xad\x90\xe5\xaf\x85\xe5\xaf\x85\xe8\xbe\xb0\xe8\xbe\xb0", 0, NULL, 0, NULL, NULL, 0, ll045, cnl_noarr, 7, 0 },
    { "生气", 1, "\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll046, 0, 3 },
    { "普护", 1, "\xe4\xb8\x91\xe5\x8d\xaf\xe7\x94\xb3\xe5\xaf\x85\xe9\x85\x89\xe5\x8d\xaf\xe6\x88\x8c\xe8\xbe\xb0\xe4\xba\xa5\xe5\xb7\xb3\xe5\xad\x90\xe5\x8d\x88", 0, NULL, 0, NULL, NULL, 0, ll036, cnl_noarr, 2, 0 },
    { "益后", 1, "\xe5\xb7\xb3\xe4\xba\xa5\xe5\xad\x90\xe5\x8d\x88\xe4\xb8\x91\xe6\x9c\xaa\xe5\xaf\x85\xe7\x94\xb3\xe5\x8d\xaf\xe9\x85\x89\xe8\xbe\xb0\xe6\x88\x8c", 0, NULL, 0, NULL, NULL, 0, ll047, cnl_noarr, 3, 0 },
    { "续世", 1, "\xe5\x8d\x88\xe5\xad\x90\xe4\xb8\x91\xe6\x9c\xaa\xe5\xaf\x85\xe7\x94\xb3\xe5\x8d\xaf\xe9\x85\x89\xe8\xbe\xb0\xe6\x88\x8c\xe5\xb7\xb3\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, ll047, cnl_noarr, 3, 0 },
    { "要安", 1, "\xe6\x9c\xaa\xe4\xb8\x91\xe5\xaf\x85\xe7\x94\xb3\xe5\x8d\xaf\xe9\x85\x89\xe8\xbe\xb0\xe6\x88\x8c\xe5\xb7\xb3\xe4\xba\xa5\xe5\x8d\x88\xe5\xad\x90", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "天后", 1, "\xe5\xaf\x85\xe4\xba\xa5\xe7\x94\xb3\xe5\xb7\xb3\xe5\xaf\x85\xe4\xba\xa5\xe7\x94\xb3\xe5\xb7\xb3\xe5\xaf\x85\xe4\xba\xa5\xe7\x94\xb3\xe5\xb7\xb3", 0, NULL, 0, NULL, NULL, 0, ll034, cnl_noarr, 1, 0 },
    { "天仓", 1, "\xe8\xbe\xb0\xe5\x8d\xaf\xe5\xaf\x85\xe4\xb8\x91\xe5\xad\x90\xe4\xba\xa5\xe6\x88\x8c\xe9\x85\x89\xe7\x94\xb3\xe6\x9c\xaa\xe5\x8d\x88\xe5\xb7\xb3", 0, NULL, 0, NULL, NULL, 0, ll048, cnl_noarr, 3, 0 },
    { "敬安", 1, "\xe5\xad\x90\xe5\x8d\x88\xe6\x9c\xaa\xe4\xb8\x91\xe7\x94\xb3\xe5\xaf\x85\xe9\x85\x89\xe5\x8d\xaf\xe6\x88\x8c\xe8\xbe\xb0\xe4\xba\xa5\xe5\xb7\xb3", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "玉宇", 1, "\xe7\x94\xb3\xe5\xaf\x85\xe5\x8d\xaf\xe9\x85\x89\xe8\xbe\xb0\xe6\x88\x8c\xe5\xb7\xb3\xe4\xba\xa5\xe5\x8d\x88\xe5\xad\x90\xe6\x9c\xaa\xe4\xb8\x91", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "金堂", 1, "\xe9\x85\x89\xe5\x8d\xaf\xe8\xbe\xb0\xe6\x88\x8c\xe5\xb7\xb3\xe4\xba\xa5\xe5\x8d\x88\xe5\xad\x90\xe6\x9c\xaa\xe4\xb8\x91\xe7\x94\xb3\xe5\xaf\x85", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "吉期", 1, "\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90", 0, NULL, 0, NULL, NULL, 0, ll049, cnl_noarr, 5, 0 },
    { "小时", 1, "\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "兵福", 1, "\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, ll050, cnl_noarr, 3, 0 },
    { "兵宝", 1, "\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90", 0, NULL, 0, NULL, NULL, 0, ll050, cnl_noarr, 3, 0 },
    { "兵吉", 6, NULL, 0, ll051, 12, NULL, NULL, 0, ll050, cnl_noarr, 3, 0 },
};

static const cnl_god_row cnl_DEMON_ROWS[100] = {
    { "岁破", 18, NULL, -1, NULL, 0, NULL, NULL, 0, cnl_noarr, ll052, 0, 4 },
    { "天罡", 1, "\xe5\x8d\xaf\xe6\x88\x8c\xe5\xb7\xb3\xe5\xad\x90\xe6\x9c\xaa\xe5\xaf\x85\xe9\x85\x89\xe8\xbe\xb0\xe4\xba\xa5\xe5\x8d\x88\xe4\xb8\x91\xe7\x94\xb3", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll012, 0, 1 },
    { "河魁", 1, "\xe9\x85\x89\xe8\xbe\xb0\xe4\xba\xa5\xe5\x8d\x88\xe4\xb8\x91\xe7\x94\xb3\xe5\x8d\xaf\xe6\x88\x8c\xe5\xb7\xb3\xe5\xad\x90\xe6\x9c\xaa\xe5\xaf\x85", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll012, 0, 1 },
    { "死神", 1, "\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll053, 0, 10 },
    { "死气", 1, "\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll054, 0, 7 },
    { "伏兵", 2, "\xe4\xb8\x99\xe7\x94\xb2\xe5\xa3\xac\xe5\xba\x9a", 3, NULL, 0, NULL, NULL, 0, cnl_noarr, ll055, 0, 3 },
    { "官符", 1, "\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll056, 0, 2 },
    { "月建", 1, "\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll057, 0, 25 },
    { "月破", 1, "\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3", 0, NULL, 0, NULL, NULL, 0, ll058, ll059, 1, 57 },
    { "月煞", 1, "\xe6\x9c\xaa\xe8\xbe\xb0\xe4\xb8\x91\xe6\x88\x8c\xe6\x9c\xaa\xe8\xbe\xb0\xe4\xb8\x91\xe6\x88\x8c\xe6\x9c\xaa\xe8\xbe\xb0\xe4\xb8\x91\xe6\x88\x8c", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll060, 0, 57 },
    { "月害", 1, "\xe6\x9c\xaa\xe5\x8d\x88\xe5\xb7\xb3\xe8\xbe\xb0\xe5\x8d\xaf\xe5\xaf\x85\xe4\xb8\x91\xe5\xad\x90\xe4\xba\xa5\xe6\x88\x8c\xe9\x85\x89\xe7\x94\xb3", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll061, 0, 27 },
    { "月刑", 1, "\xe5\x8d\xaf\xe6\x88\x8c\xe5\xb7\xb3\xe5\xad\x90\xe8\xbe\xb0\xe7\x94\xb3\xe5\x8d\x88\xe4\xb8\x91\xe5\xaf\x85\xe9\x85\x89\xe6\x9c\xaa\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll062, 0, 59 },
    { "月厌", 1, "\xe5\xad\x90\xe4\xba\xa5\xe6\x88\x8c\xe9\x85\x89\xe7\x94\xb3\xe6\x9c\xaa\xe5\x8d\x88\xe5\xb7\xb3\xe8\xbe\xb0\xe5\x8d\xaf\xe5\xaf\x85\xe4\xb8\x91", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll063, 0, 62 },
    { "月忌", 10, NULL, -1, NULL, 0, pi000, NULL, 3, cnl_noarr, ll064, 0, 2 },
    { "月虚", 1, "\xe6\x9c\xaa\xe8\xbe\xb0\xe4\xb8\x91\xe6\x88\x8c\xe6\x9c\xaa\xe8\xbe\xb0\xe4\xb8\x91\xe6\x88\x8c\xe6\x9c\xaa\xe8\xbe\xb0\xe4\xb8\x91\xe6\x88\x8c", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll065, 0, 3 },
    { "灾煞", 1, "\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll062, 0, 59 },
    { "劫煞", 1, "\xe5\xb7\xb3\xe5\xaf\x85\xe4\xba\xa5\xe7\x94\xb3\xe5\xb7\xb3\xe5\xaf\x85\xe4\xba\xa5\xe7\x94\xb3\xe5\xb7\xb3\xe5\xaf\x85\xe4\xba\xa5\xe7\x94\xb3", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll062, 0, 59 },
    { "厌对", 1, "\xe5\x8d\x88\xe5\xb7\xb3\xe8\xbe\xb0\xe5\x8d\xaf\xe5\xaf\x85\xe4\xb8\x91\xe5\xad\x90\xe4\xba\xa5\xe6\x88\x8c\xe9\x85\x89\xe7\x94\xb3\xe6\x9c\xaa", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll004, 0, 1 },
    { "招摇", 1, "\xe5\x8d\x88\xe5\xb7\xb3\xe8\xbe\xb0\xe5\x8d\xaf\xe5\xaf\x85\xe4\xb8\x91\xe5\xad\x90\xe4\xba\xa5\xe6\x88\x8c\xe9\x85\x89\xe7\x94\xb3\xe6\x9c\xaa", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll066, 0, 2 },
    { "小红砂", 1, "\xe9\x85\x89\xe4\xb8\x91\xe5\xb7\xb3\xe9\x85\x89\xe4\xb8\x91\xe5\xb7\xb3\xe9\x85\x89\xe4\xb8\x91\xe5\xb7\xb3\xe9\x85\x89\xe4\xb8\x91\xe5\xb7\xb3", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll004, 0, 1 },
    { "往亡", 1, "\xe6\x88\x8c\xe4\xb8\x91\xe5\xaf\x85\xe5\xb7\xb3\xe7\x94\xb3\xe4\xba\xa5\xe5\x8d\xaf\xe5\x8d\x88\xe9\x85\x89\xe5\xad\x90\xe8\xbe\xb0\xe6\x9c\xaa", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll067, 0, 18 },
    { "重丧", 1, "\xe7\x99\xb8\xe5\xb7\xb1\xe7\x94\xb2\xe4\xb9\x99\xe5\xb7\xb1\xe4\xb8\x99\xe4\xb8\x81\xe5\xb7\xb1\xe5\xba\x9a\xe8\xbe\x9b\xe5\xb7\xb1\xe5\xa3\xac", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll068, 0, 2 },
    { "重复", 1, "\xe7\x99\xb8\xe5\xb7\xb1\xe5\xba\x9a\xe8\xbe\x9b\xe5\xb7\xb1\xe5\xa3\xac\xe7\x99\xb8\xe6\x88\x8a\xe7\x94\xb2\xe4\xb9\x99\xe5\xb7\xb1\xe5\xa3\xac", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll068, 0, 2 },
    { "杨公忌", 12, NULL, -1, NULL, 0, pi001, NULL, 13, cnl_noarr, ll069, 0, 4 },
    { "神号", 1, "\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "妨择", 1, "\xe8\xbe\xb0\xe8\xbe\xb0\xe5\x8d\x88\xe5\x8d\x88\xe7\x94\xb3\xe7\x94\xb3\xe6\x88\x8c\xe6\x88\x8c\xe5\xad\x90\xe5\xad\x90\xe5\xaf\x85\xe5\xaf\x85", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "披麻", 1, "\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll070, 0, 2 },
    { "大耗", 1, "\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll071, 0, 5 },
    { "大祸", 2, "\xe4\xb8\x81\xe4\xb9\x99\xe7\x99\xb8\xe8\xbe\x9b", 3, NULL, 0, NULL, NULL, 0, cnl_noarr, ll055, 0, 3 },
    { "天吏", 1, "\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll072, 0, 37 },
    { "天瘟", 1, "\xe4\xb8\x91\xe5\x8d\xaf\xe6\x9c\xaa\xe6\x88\x8c\xe8\xbe\xb0\xe5\xaf\x85\xe5\x8d\x88\xe5\xad\x90\xe9\x85\x89\xe7\x94\xb3\xe5\xb7\xb3\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll073, 0, 3 },
    { "天狱", 1, "\xe5\x8d\x88\xe9\x85\x89\xe5\xad\x90\xe5\x8d\xaf\xe5\x8d\x88\xe9\x85\x89\xe5\xad\x90\xe5\x8d\xaf\xe5\x8d\x88\xe9\x85\x89\xe5\xad\x90\xe5\x8d\xaf", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "天火", 1, "\xe5\x8d\x88\xe9\x85\x89\xe5\xad\x90\xe5\x8d\xaf\xe5\x8d\x88\xe9\x85\x89\xe5\xad\x90\xe5\x8d\xaf\xe5\x8d\x88\xe9\x85\x89\xe5\xad\x90\xe5\x8d\xaf", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll074, 0, 1 },
    { "天棒", 1, "\xe5\xaf\x85\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb3\xe6\x88\x8c\xe5\xad\x90\xe5\xaf\x85\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb3\xe6\x88\x8c\xe5\xad\x90", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "天狗", 1, "\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll075, 0, 1 },
    { "天狗下食", 1, "\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll075, 0, 1 },
    { "天贼", 1, "\xe5\x8d\xaf\xe5\xaf\x85\xe4\xb8\x91\xe5\xad\x90\xe4\xba\xa5\xe6\x88\x8c\xe9\x85\x89\xe7\x94\xb3\xe6\x9c\xaa\xe5\x8d\x88\xe5\xb7\xb3\xe8\xbe\xb0", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll076, 0, 3 },
    { "地囊", 9, NULL, 0, ll078, 12, NULL, NULL, 0, cnl_noarr, ll077, 0, 16 },
    { "地火", 1, "\xe5\xad\x90\xe4\xba\xa5\xe6\x88\x8c\xe9\x85\x89\xe7\x94\xb3\xe6\x9c\xaa\xe5\x8d\x88\xe5\xb7\xb3\xe8\xbe\xb0\xe5\x8d\xaf\xe5\xaf\x85\xe4\xb8\x91", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll079, 0, 1 },
    { "独火", 1, "\xe6\x9c\xaa\xe5\x8d\x88\xe5\xb7\xb3\xe8\xbe\xb0\xe5\x8d\xaf\xe5\xaf\x85\xe4\xb8\x91\xe5\xad\x90\xe4\xba\xa5\xe6\x88\x8c\xe9\x85\x89\xe7\x94\xb3", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll000, 0, 1 },
    { "受死", 1, "\xe5\x8d\xaf\xe9\x85\x89\xe6\x88\x8c\xe8\xbe\xb0\xe4\xba\xa5\xe5\xb7\xb3\xe5\xad\x90\xe5\x8d\x88\xe4\xb8\x91\xe6\x9c\xaa\xe5\xaf\x85\xe7\x94\xb3", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll080, 0, 1 },
    { "黄沙", 1, "\xe5\xaf\x85\xe5\xad\x90\xe5\x8d\x88\xe5\xaf\x85\xe5\xad\x90\xe5\x8d\x88\xe5\xaf\x85\xe5\xad\x90\xe5\x8d\x88\xe5\xaf\x85\xe5\xad\x90\xe5\x8d\x88", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll081, 0, 1 },
    { "六不成", 1, "\xe5\x8d\xaf\xe6\x9c\xaa\xe5\xaf\x85\xe5\x8d\x88\xe6\x88\x8c\xe5\xb7\xb3\xe9\x85\x89\xe4\xb8\x91\xe7\x94\xb3\xe5\xad\x90\xe8\xbe\xb0\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll000, 0, 1 },
    { "小耗", 1, "\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll071, 0, 5 },
    { "神隔", 1, "\xe9\x85\x89\xe6\x9c\xaa\xe5\xb7\xb3\xe5\x8d\xaf\xe4\xb8\x91\xe4\xba\xa5\xe9\x85\x89\xe6\x9c\xaa\xe5\xb7\xb3\xe5\x8d\xaf\xe4\xb8\x91\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll036, 0, 2 },
    { "朱雀", 1, "\xe4\xba\xa5\xe4\xb8\x91\xe5\x8d\xaf\xe5\xb7\xb3\xe6\x9c\xaa\xe9\x85\x89\xe4\xba\xa5\xe4\xb8\x91\xe5\x8d\xaf\xe5\xb7\xb3\xe6\x9c\xaa\xe9\x85\x89", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll004, 0, 1 },
    { "白虎", 1, "\xe5\xaf\x85\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb3\xe6\x88\x8c\xe5\xad\x90\xe5\xaf\x85\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb3\xe6\x88\x8c\xe5\xad\x90", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll012, 0, 1 },
    { "玄武", 1, "\xe5\xb7\xb3\xe6\x9c\xaa\xe9\x85\x89\xe4\xba\xa5\xe4\xb8\x91\xe5\x8d\xaf\xe5\xb7\xb3\xe6\x9c\xaa\xe9\x85\x89\xe4\xba\xa5\xe4\xb8\x91\xe5\x8d\xaf", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll012, 0, 1 },
    { "勾陈", 1, "\xe6\x9c\xaa\xe9\x85\x89\xe4\xba\xa5\xe4\xb8\x91\xe5\x8d\xaf\xe5\xb7\xb3\xe6\x9c\xaa\xe9\x85\x89\xe4\xba\xa5\xe4\xb8\x91\xe5\x8d\xaf\xe5\xb7\xb3", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "木马", 1, "\xe8\xbe\xb0\xe5\x8d\x88\xe5\xb7\xb3\xe6\x9c\xaa\xe9\x85\x89\xe7\x94\xb3\xe6\x88\x8c\xe5\xad\x90\xe4\xba\xa5\xe4\xb8\x91\xe5\x8d\xaf\xe5\xaf\x85", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "破败", 1, "\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb3\xe6\x88\x8c\xe5\xad\x90\xe5\xaf\x85\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb3\xe6\x88\x8c\xe5\xad\x90\xe5\xaf\x85", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "殃败", 1, "\xe5\xb7\xb3\xe8\xbe\xb0\xe5\x8d\xaf\xe5\xaf\x85\xe4\xb8\x91\xe5\xad\x90\xe4\xba\xa5\xe6\x88\x8c\xe9\x85\x89\xe7\x94\xb3\xe6\x9c\xaa\xe5\x8d\x88", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "雷公", 1, "\xe5\xb7\xb3\xe7\x94\xb3\xe5\xaf\x85\xe4\xba\xa5\xe5\xb7\xb3\xe7\x94\xb3\xe5\xaf\x85\xe4\xba\xa5\xe5\xb7\xb3\xe7\x94\xb3\xe5\xaf\x85\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "飞廉", 1, "\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll082, 0, 4 },
    { "大煞", 1, "\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll050, 0, 3 },
    { "枯鱼", 1, "\xe7\x94\xb3\xe5\xb7\xb3\xe8\xbe\xb0\xe4\xb8\x91\xe6\x88\x8c\xe6\x9c\xaa\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\xaf\x85\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll079, 0, 1 },
    { "九空", 1, "\xe7\x94\xb3\xe5\xb7\xb3\xe8\xbe\xb0\xe4\xb8\x91\xe6\x88\x8c\xe6\x9c\xaa\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\xaf\x85\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll083, 0, 6 },
    { "八座", 1, "\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "八风触水龙", 9, NULL, 1, ll084, 4, NULL, NULL, 0, cnl_noarr, ll066, 0, 2 },
    { "血忌", 1, "\xe5\x8d\x88\xe5\xad\x90\xe4\xb8\x91\xe6\x9c\xaa\xe5\xaf\x85\xe7\x94\xb3\xe5\x8d\xaf\xe9\x85\x89\xe8\xbe\xb0\xe6\x88\x8c\xe5\xb7\xb3\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll085, 0, 1 },
    { "阴错", 14, "\xe5\xa3\xac\xe5\xad\x90\xe7\x99\xb8\xe4\xb8\x91\xe5\xba\x9a\xe5\xaf\x85\xe8\xbe\x9b\xe5\x8d\xaf\xe5\xba\x9a\xe8\xbe\xb0\xe4\xb8\x81\xe5\xb7\xb3\xe4\xb8\x99\xe5\x8d\x88\xe4\xb8\x81\xe6\x9c\xaa\xe7\x94\xb2\xe7\x94\xb3\xe4\xb9\x99\xe9\x85\x89\xe7\x94\xb2\xe6\x88\x8c\xe7\x99\xb8\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "三娘煞", 10, NULL, -1, NULL, 0, pi002, NULL, 6, cnl_noarr, ll086, 0, 2 },
    { "四绝", 11, NULL, -1, NULL, 0, NULL, NULL, 0, cnl_noarr, ll087, 0, 8 },
    { "四离", 11, NULL, -1, NULL, 0, NULL, NULL, 0, cnl_noarr, ll088, 0, 2 },
    { "四击", 1, "\xe6\x9c\xaa\xe6\x9c\xaa\xe6\x88\x8c\xe6\x88\x8c\xe6\x88\x8c\xe4\xb8\x91\xe4\xb8\x91\xe4\xb8\x91\xe8\xbe\xb0\xe8\xbe\xb0\xe8\xbe\xb0\xe6\x9c\xaa", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll050, 0, 3 },
    { "四耗", 9, NULL, 1, ll090, 4, NULL, NULL, 0, cnl_noarr, ll089, 0, 8 },
    { "四穷", 9, NULL, 1, ll092, 4, NULL, NULL, 0, cnl_noarr, ll091, 0, 13 },
    { "四忌", 9, NULL, 1, ll094, 4, NULL, NULL, 0, cnl_noarr, ll093, 0, 7 },
    { "四废", 9, NULL, 1, ll096, 4, NULL, NULL, 0, cnl_noarr, ll095, 0, 55 },
    { "五墓", 9, NULL, 0, ll098, 12, NULL, NULL, 0, cnl_noarr, ll097, 0, 30 },
    { "五虚", 6, NULL, 1, ll100, 4, NULL, NULL, 0, cnl_noarr, ll099, 0, 2 },
    { "五离", 7, "\xe7\x94\xb3\xe9\x85\x89", -1, NULL, 0, NULL, NULL, 0, ll101, ll102, 1, 5 },
    { "五鬼", 1, "\xe6\x9c\xaa\xe6\x88\x8c\xe5\x8d\x88\xe5\xaf\x85\xe8\xbe\xb0\xe9\x85\x89\xe5\x8d\xaf\xe7\x94\xb3\xe4\xb8\x91\xe5\xb7\xb3\xe5\xad\x90\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll081, 0, 1 },
    { "八专", 8, NULL, -1, ll104, 5, NULL, NULL, 0, cnl_noarr, ll103, 0, 6 },
    { "九坎", 1, "\xe7\x94\xb3\xe5\xb7\xb3\xe8\xbe\xb0\xe4\xb8\x91\xe6\x88\x8c\xe6\x9c\xaa\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\xaf\x85\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll105, 0, 4 },
    { "九焦", 1, "\xe7\x94\xb3\xe5\xb7\xb3\xe8\xbe\xb0\xe4\xb8\x91\xe6\x88\x8c\xe6\x9c\xaa\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\xaf\x85\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll106, 0, 2 },
    { "天转", 14, "\xe4\xb9\x99\xe5\x8d\xaf\xe4\xb8\x99\xe5\x8d\x88\xe8\xbe\x9b\xe9\x85\x89\xe5\xa3\xac\xe5\xad\x90", 1, NULL, 0, NULL, NULL, 0, cnl_noarr, ll107, 0, 3 },
    { "地转", 14, "\xe8\xbe\x9b\xe5\x8d\xaf\xe6\x88\x8a\xe5\x8d\x88\xe7\x99\xb8\xe9\x85\x89\xe4\xb8\x99\xe5\xad\x90", 1, NULL, 0, NULL, NULL, 0, cnl_noarr, ll107, 0, 3 },
    { "月建转杀", 1, "\xe5\x8d\xaf\xe5\x8d\x88\xe9\x85\x89\xe5\xad\x90", 1, NULL, 0, NULL, NULL, 0, cnl_noarr, ll000, 0, 1 },
    { "荒芜", 15, "\xe5\xb7\xb3\xe9\x85\x89\xe4\xb8\x91\xe7\x94\xb3\xe5\xad\x90\xe8\xbe\xb0\xe4\xba\xa5\xe5\x8d\xaf\xe6\x9c\xaa\xe5\xaf\x85\xe5\x8d\x88\xe6\x88\x8c", 1, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "蚩尤", 1, "\xe6\x88\x8c\xe5\xad\x90\xe5\xaf\x85\xe8\xbe\xb0\xe5\x8d\x88\xe7\x94\xb3", 4, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "大时", 1, "\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll072, 0, 37 },
    { "大败", 1, "\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "咸池", 1, "\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90\xe9\x85\x89\xe5\x8d\x88\xe5\x8d\xaf\xe5\xad\x90", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll108, 0, 3 },
    { "土符", 1, "\xe7\x94\xb3\xe5\xad\x90\xe4\xb8\x91\xe5\xb7\xb3\xe9\x85\x89\xe5\xaf\x85\xe5\x8d\x88\xe6\x88\x8c\xe5\x8d\xaf\xe6\x9c\xaa\xe4\xba\xa5\xe8\xbe\xb0", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll077, 0, 16 },
    { "土府", 1, "\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c\xe4\xba\xa5", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll077, 0, 16 },
    { "土王用事", 16, NULL, -1, NULL, 0, NULL, NULL, 0, cnl_noarr, ll077, 0, 16 },
    { "血支", 1, "\xe4\xba\xa5\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\x8d\xaf\xe8\xbe\xb0\xe5\xb7\xb3\xe5\x8d\x88\xe6\x9c\xaa\xe7\x94\xb3\xe9\x85\x89\xe6\x88\x8c", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll085, 0, 1 },
    { "游祸", 1, "\xe4\xba\xa5\xe7\x94\xb3\xe5\xb7\xb3\xe5\xaf\x85\xe4\xba\xa5\xe7\x94\xb3\xe5\xb7\xb3\xe5\xaf\x85\xe4\xba\xa5\xe7\x94\xb3\xe5\xb7\xb3\xe5\xaf\x85", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll109, 0, 4 },
    { "归忌", 1, "\xe5\xaf\x85\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\xad\x90\xe4\xb8\x91\xe5\xaf\x85\xe5\xad\x90\xe4\xb8\x91", 0, NULL, 0, NULL, NULL, 0, cnl_noarr, ll110, 0, 2 },
    { "岁薄", 13, NULL, -1, NULL, 0, pi003, ll111, 4, cnl_noarr, cnl_noarr, 0, 0 },
    { "逐阵", 13, NULL, -1, NULL, 0, pi004, ll111, 4, cnl_noarr, cnl_noarr, 0, 0 },
    { "阴阳交破", 13, NULL, -1, NULL, 0, pi005, ll112, 1, cnl_noarr, cnl_noarr, 0, 0 },
    { "宝日", 8, NULL, -1, ll113, 12, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "义日", 8, NULL, -1, ll114, 12, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "制日", 8, NULL, -1, ll115, 12, NULL, NULL, 0, cnl_noarr, cnl_noarr, 0, 0 },
    { "伐日", 8, NULL, -1, ll116, 12, NULL, NULL, 0, cnl_noarr, ll050, 0, 3 },
    { "专日", 8, NULL, -1, ll117, 12, NULL, NULL, 0, cnl_noarr, ll050, 0, 3 },
    { "重日", 7, "\xe5\xb7\xb3\xe4\xba\xa5", -1, NULL, 0, NULL, NULL, 0, cnl_noarr, ll118, 0, 3 },
    { "复日", 1, "\xe7\x99\xb8\xe5\xb7\xb3\xe7\x94\xb2\xe4\xb9\x99\xe6\x88\x8a\xe4\xb8\x99\xe4\xb8\x81\xe5\xb7\xb3\xe5\xba\x9a\xe8\xbe\x9b\xe6\x88\x8a\xe5\xa3\xac", 0, NULL, 0, NULL, NULL, 0, ll119, ll118, 1, 3 },
};

/* ==== officer things ==== */
typedef struct { const char *ch; const char *const *good; int n_good; const char *const *bad; int n_bad; } cnl_officer_t;
static const cnl_officer_t cnl_officer_things[12] = {
    { "\xe5\xbb\xba", ll120, 6, cnl_noarr, 0 },
    { "\xe9\x99\xa4", ll044, 7, cnl_noarr, 0 },
    { "\xe6\xbb\xa1", ll121, 10, ll122, 8 },
    { "\xe5\xb9\xb3", ll123, 2, ll124, 51 },
    { "\xe5\xae\x9a", ll125, 1, cnl_noarr, 0 },
    { "\xe6\x89\xa7", ll126, 1, cnl_noarr, 0 },
    { "\xe7\xa0\xb4", ll034, 1, cnl_noarr, 0 },
    { "\xe5\x8d\xb1", ll127, 3, cnl_noarr, 0 },
    { "\xe6\x88\x90", ll128, 5, cnl_noarr, 0 },
    { "\xe6\x94\xb6", ll129, 4, ll130, 45 },
    { "\xe5\xbc\x80", ll131, 34, cnl_noarr, 0 },
    { "\xe9\x97\xad", ll132, 3, ll133, 31 },
};

/* ==== day8 char thing ==== */
typedef struct { const char *ch; const char *const *good; int n_good; const char *const *bad; int n_bad; } cnl_d8thing_t;
static const cnl_d8thing_t cnl_day8_char_thing[16] = {
    { "\xe7\x94\xb2", cnl_noarr, 0, ll134, 1 },
    { "\xe4\xb9\x99", cnl_noarr, 0, ll079, 1 },
    { "\xe4\xb8\x81", cnl_noarr, 0, ll135, 2 },
    { "\xe5\xba\x9a", cnl_noarr, 0, ll136, 1 },
    { "\xe8\xbe\x9b", cnl_noarr, 0, ll137, 1 },
    { "\xe5\xa3\xac", cnl_noarr, 0, ll138, 2 },
    { "\xe5\xad\x90", ll101, 1, cnl_noarr, 0 },
    { "\xe4\xb8\x91", cnl_noarr, 0, ll125, 1 },
    { "\xe5\xaf\x85", cnl_noarr, 0, ll075, 1 },
    { "\xe5\x8d\xaf", cnl_noarr, 0, ll139, 1 },
    { "\xe9\x85\x89", cnl_noarr, 0, ll140, 1 },
    { "\xe5\xb7\xb3", cnl_noarr, 0, ll081, 1 },
    { "\xe5\x8d\x88", cnl_noarr, 0, ll074, 1 },
    { "\xe6\x9c\xaa", cnl_noarr, 0, ll034, 1 },
    { "\xe7\x94\xb3", cnl_noarr, 0, ll141, 1 },
    { "\xe4\xba\xa5", ll101, 1, ll004, 1 },
};

/* ==== thing level ==== */
typedef struct { const char *pat; const char *const *gods; int n; int level; } cnl_tl_sub;
typedef struct { const char *key; const cnl_tl_sub *sub; int n; } cnl_tl_entry;
static const cnl_tl_sub cnl_tl_00[7] = {
    { "\xe4\xba\xa5", ll142, 3, 0 },
    { "\xe5\xb7\xb3", ll143, 3, 1 },
    { "\xe7\x94\xb3", ll144, 2, 2 },
    { "\xe5\xaf\x85", ll145, 3, 3 },
    { "\xe5\x8d\xaf\xe5\x8d\x88\xe9\x85\x89", ll146, 1, 3 },
    { "\xe8\xbe\xb0\xe6\x88\x8c\xe4\xb8\x91\xe6\x9c\xaa", ll147, 1, 4 },
    { "\xe5\xad\x90", ll148, 2, 4 },
};
static const cnl_tl_sub cnl_tl_01[6] = {
    { "\xe5\xaf\x85\xe7\x94\xb3", ll149, 3, 0 },
    { "\xe5\xb7\xb3\xe4\xba\xa5", ll150, 2, 2 },
    { "\xe8\xbe\xb0\xe6\x9c\xaa", ll151, 1, 2 },
    { "\xe5\xad\x90\xe5\x8d\x88\xe9\x85\x89", ll152, 1, 3 },
    { "\xe4\xb8\x91\xe6\x88\x8c", ll153, 1, 3 },
    { "\xe5\x8d\xaf", ll152, 1, 4 },
};
static const cnl_tl_sub cnl_tl_02[3] = {
    { "\xe5\xad\x90\xe5\x8d\x88\xe5\x8d\xaf\xe9\x85\x89", ll154, 1, 3 },
    { "\xe8\xbe\xb0\xe6\x88\x8c\xe4\xb8\x91\xe6\x9c\xaa", ll155, 2, 3 },
    { "\xe5\xaf\x85\xe7\x94\xb3\xe5\xb7\xb3\xe4\xba\xa5", ll147, 1, 4 },
};
static const cnl_tl_sub cnl_tl_03[4] = {
    { "\xe5\xaf\x85\xe7\x94\xb3", ll156, 2, 0 },
    { "\xe8\xbe\xb0\xe6\x88\x8c\xe4\xb8\x91\xe6\x9c\xaa", ll157, 2, 1 },
    { "\xe5\xb7\xb3\xe4\xba\xa5", ll158, 2, 2 },
    { "\xe5\xad\x90\xe5\x8d\x88\xe5\x8d\xaf\xe9\x85\x89", ll159, 1, 3 },
};
static const cnl_tl_sub cnl_tl_04[4] = {
    { "\xe5\xaf\x85\xe7\x94\xb3\xe5\xb7\xb3\xe4\xba\xa5", ll160, 1, 1 },
    { "\xe8\xbe\xb0\xe6\x88\x8c\xe4\xb8\x91\xe6\x9c\xaa", ll161, 2, 2 },
    { "\xe5\xad\x90\xe5\x8d\x88", ll162, 1, 4 },
    { "\xe5\x8d\xaf\xe9\x85\x89", ll163, 2, 5 },
};
static const cnl_tl_sub cnl_tl_05[2] = {
    { "\xe5\x8d\xaf\xe9\x85\x89", ll164, 2, 1 },
    { "\xe5\xad\x90\xe5\x8d\x88", ll165, 2, 3 },
};
static const cnl_tl_sub cnl_tl_06[7] = {
    { "\xe5\xb7\xb3", ll166, 3, 1 },
    { "\xe5\xaf\x85", ll167, 3, 3 },
    { "\xe8\xbe\xb0\xe9\x85\x89\xe4\xba\xa5", ll168, 1, 3 },
    { "\xe5\xad\x90", ll169, 2, 4 },
    { "\xe5\x8d\xaf", ll170, 3, 4 },
    { "\xe6\x9c\xaa\xe7\x94\xb3", ll162, 1, 4 },
    { "\xe5\x8d\x88", ll171, 3, 4 },
};
static const cnl_tl_sub cnl_tl_07[7] = {
    { "\xe5\x8d\xaf\xe9\x85\x89", ll172, 2, 2 },
    { "\xe4\xb8\x91\xe6\x9c\xaa", ll173, 2, 2 },
    { "\xe5\xb7\xb3\xe4\xba\xa5", ll150, 2, 2 },
    { "\xe7\x94\xb3", ll174, 2, 2 },
    { "\xe5\xad\x90\xe5\x8d\x88", ll147, 1, 3 },
    { "\xe8\xbe\xb0\xe6\x88\x8c", ll175, 3, 3 },
    { "\xe5\xaf\x85", ll176, 3, 3 },
};
static const cnl_tl_sub cnl_tl_08[7] = {
    { "\xe5\xaf\x85\xe7\x94\xb3", ll177, 1, 2 },
    { "\xe4\xb8\x91\xe6\x9c\xaa", ll160, 1, 2 },
    { "\xe8\xbe\xb0\xe6\x88\x8c", ll178, 1, 3 },
    { "\xe5\xb7\xb3\xe4\xba\xa5", ll179, 1, 3 },
    { "\xe5\xad\x90", ll180, 2, 4 },
    { "\xe5\x8d\x88", ll181, 3, 4 },
    { "\xe5\x8d\xaf\xe9\x85\x89", ll182, 2, 5 },
};
static const cnl_tl_sub cnl_tl_09[5] = {
    { "\xe5\xaf\x85\xe7\x94\xb3\xe5\xb7\xb3\xe4\xba\xa5", ll183, 2, 0 },
    { "\xe8\xbe\xb0\xe6\x88\x8c", ll184, 2, 0 },
    { "\xe4\xb8\x91\xe6\x9c\xaa", ll185, 2, 2 },
    { "\xe5\xad\x90\xe5\x8d\x88\xe9\x85\x89", ll186, 1, 3 },
    { "\xe5\x8d\xaf", ll187, 2, 4 },
};
static const cnl_tl_sub cnl_tl_10[4] = {
    { "\xe5\xaf\x85\xe7\x94\xb3\xe5\xb7\xb3\xe4\xba\xa5", ll188, 1, 2 },
    { "\xe8\xbe\xb0\xe6\x88\x8c\xe4\xb8\x91\xe6\x9c\xaa", ll189, 1, 3 },
    { "\xe5\x8d\xaf\xe5\x8d\x88\xe9\x85\x89", ll190, 1, 3 },
    { "\xe5\xad\x90", ll191, 2, 4 },
};
static const cnl_tl_entry cnl_thing_level[11] = {
    { "\xe5\xb9\xb3\xe6\x97\xa5", cnl_tl_00, 7 },
    { "\xe6\x94\xb6\xe6\x97\xa5", cnl_tl_01, 6 },
    { "\xe9\x97\xad\xe6\x97\xa5", cnl_tl_02, 3 },
    { "\xe5\x8a\xab\xe7\x85\x9e", cnl_tl_03, 4 },
    { "\xe7\x81\xbe\xe7\x85\x9e", cnl_tl_04, 4 },
    { "\xe6\x9c\x88\xe7\x85\x9e", cnl_tl_05, 2 },
    { "\xe6\x9c\x88\xe5\x88\x91", cnl_tl_06, 7 },
    { "\xe6\x9c\x88\xe5\xae\xb3", cnl_tl_07, 7 },
    { "\xe6\x9c\x88\xe5\x8e\x8c", cnl_tl_08, 7 },
    { "\xe5\xa4\xa7\xe6\x97\xb6", cnl_tl_09, 5 },
    { "\xe5\xa4\xa9\xe5\x90\x8f", cnl_tl_10, 4 },
};


/* ====================== LUNAR DATE CONVERSION ====================== */

static int cnl_abs(int x) { return x < 0 ? -x : x; }

/* fills R->month_days from current lunar_year/lunar_month; mirrors
   getMonthLeapMonthLeapDays() */
static int cnl_month_leap_days(cnlunar_result *R) {
    int idx = R->lunar_year - 1901;
    unsigned int tmp;
    /* mirror Python negative-index wraparound (table[-1] == last entry) */
    if (idx < -1 || idx >= 199) return CNLUNAR_ERR_RANGE;
    if (idx < 0) idx = 198;
    tmp = cnl_LUNAR_MONTH[idx];
    R->month_days[0] = (tmp & (1u << (R->lunar_month - 1))) ? 30 : 29;
    int leap_month = (int)((tmp >> 13) & 0xf);
    R->month_days[1] = leap_month;
    R->month_days[2] = leap_month ? (((tmp & (1u << 12)) ? 30 : 29)) : 0;
    return CNLUNAR_OK;
}

/* mirrors get_lunarDateNum(); sets lunar_year/month/day, span_days,
   is_leap_month, month_days. Returns error code. */
static int cnl_lunar_date_num(cnlunar_result *R) {
    R->lunar_year = R->year; R->lunar_month = 1; R->lunar_day = 1;
    R->is_leap_month = 0;
    if (R->lunar_year < 1901 || R->lunar_year > 2100) return CNLUNAR_ERR_RANGE;
    unsigned int code = cnl_NEWYEAR[R->lunar_year - 1901];
    int ny_m = (int)((code >> 5) & 0x3);
    int ny_d = (int)(code & 0x1f);
    int ref = cnl_daynum(R->lunar_year, ny_m, ny_d);
    int span = cnl_daynum(R->year, R->month, R->day) - ref;
    R->span_days = span;

    if (span >= 0) {
        int rc = cnl_month_leap_days(R);
        if (rc) return rc;
        int month_days = R->month_days[0], leap_month = R->month_days[1], leap_day = R->month_days[2];
        while (span >= month_days) {
            span -= month_days;
            if (R->lunar_month == leap_month) {
                month_days = leap_day;
                if (span < month_days) { R->is_leap_month = 1; break; }
                span -= month_days;
            }
            R->lunar_month += 1;
            rc = cnl_month_leap_days(R);
            if (rc) return rc;
            month_days = R->month_days[0];
        }
        R->lunar_day += span;
        return CNLUNAR_OK;
    } else {
        R->lunar_month = 12;
        R->lunar_year -= 1;
        int rc = cnl_month_leap_days(R);
        if (rc) return rc;
        int month_days = R->month_days[0], leap_month = R->month_days[1], leap_day = R->month_days[2];
        while (cnl_abs(span) > month_days) {
            span += month_days;
            R->lunar_month -= 1;
            if (R->lunar_month == leap_month) {
                month_days = leap_day;
                if (cnl_abs(span) <= month_days) { R->is_leap_month = 1; break; }
                span += month_days;
            }
            rc = cnl_month_leap_days(R);
            if (rc) return rc;
            month_days = R->month_days[0];
        }
        R->lunar_day += (month_days + span);
        return CNLUNAR_OK;
    }
}

/* ------------------ 60-cycle helpers ------------------ */
static void cnl_g60_at(char out[7], int k) {
    k = cnl_mod(k, 60);
    cnl_cpy(out, cnl_G60[k], 7);
}
/* stem index (0..9) of the stem char of an 8char string */
static int cnl_stem_num_of8(const char *s8) {
    for (int i = 0; i < 10; ++i) if (memcmp(s8, cnl_STEM10_BASE + 3 * i, 3) == 0) return i;
    return 0;
}
static int cnl_branch_num_of8(const char *s8) {
    for (int i = 0; i < 12; ++i) if (memcmp(s8 + 3, cnl_BRANCH12_BASE + 3 * i, 3) == 0) return i;
    return 0;
}

/* ------------------ 农历中文字符串 ------------------ */
static void cnl_lunar_year_cn(cnlunar_result *R) {
    int y = R->lunar_year;
    int div = 1000;
    int w = 0;
    for (int p = 0; p < 4; ++p) {
        int digit = (y / div) % 10;
        cnl_cpy(R->lunar_year_cn + w * 3, cnl_UPPER_NUM[digit], 4);
        w += 1;
        div /= 10;
    }
    R->lunar_year_cn[w * 3] = 0;
}

static void cnl_lunar_month_cn(cnlunar_result *R) {
    const char *nm = cnl_MONTH_NAME[(R->lunar_month - 1) % 12];
    int this_days = R->month_days[0];
    if (R->is_leap_month) this_days = R->month_days[2];
    R->lunar_month_long = (this_days >= 30) ? 1 : 0;
    char base[CNLUNAR_ITEM_MAX];
    char tmp[CNLUNAR_ITEM_MAX];
    cnl_cpy(base, nm, CNLUNAR_ITEM_MAX);
    if (R->is_leap_month) {
        cnl_cpy(tmp, "\xe9\x97\xb0", 4);           /* 闰 */
        int _lb=(int)strlen(base); memcpy(tmp+3, base, _lb); tmp[3+_lb]=0;
        cnl_cpy(base, tmp, CNLUNAR_ITEM_MAX);
    }
    size_t blen = strlen(base);
    if (blen + 3 < CNLUNAR_ITEM_MAX) {
        if (R->lunar_month_long) { base[blen] = (char)0xe5; base[blen + 1] = (char)0xa4; base[blen + 2] = (char)0xa7; base[blen + 3] = 0; }
        else                    { base[blen] = (char)0xe5; base[blen + 1] = (char)0xb0; base[blen + 2] = (char)0x8f; base[blen + 3] = 0; }
    }
    cnl_cpy(R->lunar_month_cn, base, CNLUNAR_ITEM_MAX);
}

static void cnl_lunar_day_cn(cnlunar_result *R) {
    cnl_cpy(R->lunar_day_cn, cnl_DAY_NAME[(R->lunar_day - 1) % 30], 8);
}

static void cnl_phase_of_moon(cnlunar_result *R) {
    R->phase_of_moon[0] = 0;
    if (R->lunar_day - R->lunar_month_long == 15) cnl_cpy(R->phase_of_moon, "\xe6\x9c\x9b", 4);
    else if (R->lunar_day == 1) cnl_cpy(R->phase_of_moon, "\xe6\x9c\x94", 4);
    else if (R->lunar_day >= 7 && R->lunar_day <= 8) cnl_cpy(R->phase_of_moon, "\xe4\xb8\x8a\xe5\xbc\xa6", 7);
    else if (R->lunar_day >= 22 && R->lunar_day <= 23) cnl_cpy(R->phase_of_moon, "\xe4\xb8\x8b\xe5\xbc\xa6", 7);
}

/* ------------------ 二十四节气 ------------------ */
static void cnl_year_solar_terms(int year, cnlunar_md out[24]) {
    unsigned long long data = cnl_SOLAR[year - 1901];
    for (int k = 0; k < 24; ++k) {
        int val = (int)((data >> (2 * k)) & 3ULL);
        out[k].month = k / 2 + 1;
        out[k].day = cnl_ENC_VECTOR[k] + val;
    }
}

static int cnlunar_md_le(cnlunar_md a, cnlunar_md b) { return a.month < b.month || (a.month == b.month && a.day <= b.day); }

static void cnl_today_solar_terms(cnlunar_result *R) {
    int year = R->year;
    cnlunar_md terms[24];
    cnl_year_solar_terms(year, terms);
    for (int i = 0; i < 24; ++i) { R->this_year_solar_terms[i].date = terms[i]; cnl_cpy(R->this_year_solar_terms[i].name, cnl_SOLAR_TERMS_NAME[i], 8); }

    cnlunar_md find = { R->month, R->day };
    int cnt = 0;
    for (int i = 0; i < 24; ++i) if (cnlunar_md_le(terms[i], find)) ++cnt;
    int next_num = cnt % 24;
    R->next_solar_num = next_num;

    int today_idx = -1;
    for (int i = 0; i < 24; ++i)
        if (terms[i].month == find.month && terms[i].day == find.day) { today_idx = i; break; }
    if (today_idx >= 0) cnl_cpy(R->today_solar_terms, cnl_SOLAR_TERMS_NAME[today_idx], 8);
    else cnl_cpy(R->today_solar_terms, "\xe6\x97\xa0", 4);   /* 无 */

    int next_year = year;
    if (terms[23].month == find.month && find.day >= terms[23].day) {
        next_year = year + 1;
        cnl_year_solar_terms(next_year, terms);
    }
    cnl_cpy(R->next_solar_term, cnl_SOLAR_TERMS_NAME[next_num], 8);
    R->next_solar_term_date = terms[next_num];
    R->next_solar_term_year = next_year;
}

/* ------------------ 八字 / 季节 / 时辰 ------------------ */
static void cnl_8char(cnlunar_result *R, int x) {
    cnl_g60_at(R->year8char, (R->lunar_year - 4) % 60 - x);
    int next_num = R->next_solar_num;
    if (next_num == 0 && R->month == 12) next_num = 24;
    int apart = (next_num + 1) / 2;
    cnl_g60_at(R->month8char, (R->year - 2019) * 12 + apart);
    int apart_days = cnl_daynum(R->year, R->month, R->day) - cnl_daynum(2019, 1, 29);
    int base = 2;                       /* 丙寅 index */
    if (R->twohour_num == 12) base += 1;
    R->day_heavenly_earth_num = cnl_mod(apart_days + base, 60);
    cnl_g60_at(R->day8char, R->day_heavenly_earth_num);
}

static void cnl_twohour(cnlunar_result *R) {
    int begin = cnl_mod(R->day_heavenly_earth_num * 12, 60);
    cnl_l_reset(&R->twohour8char_list);
    for (int i = 0; i < 13; ++i) cnl_l_add(&R->twohour8char_list, cnl_G60[cnl_mod(begin + i, 60)]);
    int hi = R->twohour_num % 12;
    cnl_cpy(R->twohour8char, R->twohour8char_list.it[hi], 8);
}

static void cnl_nums_and_season(cnlunar_result *R) {
    R->year_earth_num = cnl_branch_num_of8(R->year8char);
    R->month_earth_num = cnl_branch_num_of8(R->month8char);
    R->day_earth_num = cnl_branch_num_of8(R->day8char);
    R->year_heaven_num = cnl_stem_num_of8(R->year8char);
    R->month_heaven_num = cnl_stem_num_of8(R->month8char);
    R->day_heaven_num = cnl_stem_num_of8(R->day8char);

    R->lunar_season_type = R->month_earth_num % 3;
    R->lunar_season_num = ((R->month_earth_num + 10) % 12) / 3;
    cnl_cpy(R->lunar_month_type, cnl_MONTH_TYPE + 3 * R->lunar_season_type, 4);
    cnl_cpy(R->lunar_season, cnl_SEASON_TYPE + 3 * R->lunar_season_num, 4);
    cnl_cpy(R->lunar_season_name, R->lunar_month_type, 4);
    {
        int w = (int)strlen(R->lunar_season_name);
        for (int i = 0; i < 3 && w < 7; ++i) R->lunar_season_name[w + i] = R->lunar_season[i];
        R->lunar_season_name[w + 3] = 0;
    }
}

static void cnl_weekday_starzodiac(cnlunar_result *R) {
    int days = cnl_daynum(R->year, R->month, R->day);
    cnl_cpy(R->week_day_cn, cnl_WEEKDAY[cnl_mod(days + 3, 7)], 12);
    /* 星座: 数通过的星座分界点 (Python: count of y <= date) */
    int sn = 0;
    cnlunar_md find = { R->month, R->day };
    for (int i = 0; i < 12; ++i) { cnlunar_md b = { cnl_STAR_ZODIAC_DATE[i][0], cnl_STAR_ZODIAC_DATE[i][1] }; if (cnlunar_md_le(b, find)) ++sn; }
    cnl_cpy(R->star_zodiac, cnl_STAR_ZODIAC[sn % 12], 16);
}

/* 立春干支偏移 (year8Char option) */
static int cnl_beginning_of_spring_x(cnlunar_result *R) {
    int x = 0;
    if (!(R->options & CNLUNAR_YEAR8CHAR_LICHUN)) return 0;
    int is_before_bos = R->next_solar_num < 3;
    int is_before_lunar_year = R->span_days < 0;
    if (is_before_lunar_year) {
        if (!is_before_bos) x = -1;
    } else {
        if (is_before_bos) x = 1;
    }
    return x;
}


/* ------------------ 建除十二神 / 值神 ------------------ */
static void cnl_12day_officer(cnlunar_result *R) {
    int men = (R->options & CNLUNAR_GODTYPE_CNLUNAR) ? (R->lunar_month - 1 + 2) % 12 : R->month_earth_num;
    int apart = R->day_earth_num - (men % 12);
    cnl_cpy(R->today12day_officer, cnl_12_OFFICER_CHARS + 3 * cnl_mod(apart, 12), 4);
    static const int ecliptic_seed[12] = {8,10,0,2,4,6,8,10,0,2,4,6};
    int ecl = cnl_mod(R->day_earth_num - ecliptic_seed[men], 12);
    cnl_cpy(R->today12day_god, cnl_12_DAYGOD[ecl], 8);
    int good = (ecl == 0 || ecl == 1 || ecl == 4 || ecl == 5 || ecl == 7 || ecl == 10);
    cnl_cpy(R->today12day_name, good ? "\xe9\xbb\x84\xe9\x81\x93\xe6\x97\xa5" : "\xe9\xbb\x91\xe9\x81\x93\xe6\x97\xa5", 12); /* 黄道日/黑道日 */
}

/* ------------------ 生肖冲煞 ------------------ */
static int cnl_begin_of_spring_x_value(cnlunar_result *R) {
    return cnl_beginning_of_spring_x(R);
}

static void cnl_zodiac(cnlunar_result *R) {
    int x = cnl_begin_of_spring_x_value(R);
    cnl_cpy(R->chinese_year_zodiac, cnl_CHINESE_ZODIAC[cnl_mod((R->lunar_year - 4) % 12 - x, 12)], 4);
    int zn = R->day_earth_num;
    int cn = (zn + 6) % 12;
    cnl_cpy(R->zodiac_mark6, cnl_CHINESE_ZODIAC[cnl_mod(25 - zn, 12)], 4);
    cnl_l_reset(&R->zodiac_mark3);
    cnl_l_push(&R->zodiac_mark3, cnl_CHINESE_ZODIAC[(zn + 4) % 12]);
    cnl_l_push(&R->zodiac_mark3, cnl_CHINESE_ZODIAC[(zn + 8) % 12]);
    cnl_cpy(R->zodiac_win, cnl_CHINESE_ZODIAC[zn], 4);
    cnl_cpy(R->zodiac_lose, cnl_CHINESE_ZODIAC[cn], 4);
    {
        char t[16];
        cnl_cpy(t, R->zodiac_win, 4);
        strncat(t, "\xe6\x97\xa5\xe5\x86\xb2", 15 - (int)strlen(t));
        strncat(t, R->zodiac_lose, 15 - (int)strlen(t));
        cnl_cpy(R->chinese_zodiac_clash, t, 16);
    }
}

/* ------------------ 廿八宿 / 纳音 / 五行 / 飞星 ------------------ */
static void cnl_28stars(cnlunar_result *R) {
    int apart = cnl_daynum(R->year, R->month, R->day) - cnl_daynum(2019, 1, 17);
    cnl_cpy(R->today28star, cnl_STAR28[cnl_mod(apart, 28)], 12);
}

/* take the LAST codepoint of a string into dst */
static void cnl_utf8_last(char *dst, const char *src) {
    const char *p = src + strlen(src);
    if (p == src) { dst[0] = 0; return; }
    --p;
    while (p > src && (((*(const unsigned char *)p) & 0xc0) == 0x80)) --p;
    int n = 1;
    while (((*(const unsigned char *)(p + n)) & 0xc0) == 0x80) ++n;
    for (int i = 0; i < n; ++i) dst[i] = p[i];
    dst[n] = 0;
}

static void cnl_5elements(cnlunar_result *R) {
    cnl_l_reset(&R->today5elements);
    char buf[CNLUNAR_ITEM_MAX];
    cnl_l_push(&R->today5elements, "\xe5\xa4\xa9\xe5\xb9\xb2");                             /* 天干 */
    { char c3[4]; cnl_cpy(c3, cnl_STEM10_BASE + 3 * cnl_stem_num_of8(R->day8char), 4); cnl_l_push(&R->today5elements, c3); } /* 天干 char */
    { char sx[4]; cnl_cpy(sx, cnl_STEM5ELEM_BASE + 3 * R->day_heaven_num, 4); cnl_cpy(buf, "\xe5\xb1\x9e", 4); memcpy(buf + 3, sx, 3); buf[6] = 0; cnl_l_push(&R->today5elements, buf); }
    cnl_l_push(&R->today5elements, "\xe5\x9c\xb0\xe6\x94\xaf");                             /* 地支 */
    { char c3[4]; cnl_cpy(c3, cnl_BRANCH12_BASE + 3 * cnl_branch_num_of8(R->day8char), 4); cnl_l_push(&R->today5elements, c3); }
    { char sx[4]; cnl_cpy(sx, cnl_BRANCH5ELEM_BASE + 3 * R->day_earth_num, 4); cnl_cpy(buf, "\xe5\xb1\x9e", 4); memcpy(buf + 3, sx, 3); buf[6] = 0; cnl_l_push(&R->today5elements, buf); }
    cnl_l_push(&R->today5elements, "\xe7\xba\xb3\xe9\x9f\xb3");                               /* 纳音 */
    {
        const char *nayin = cnl_NAYIN30[R->day_heavenly_earth_num / 2];
        char last[4];
        cnl_utf8_last(last, nayin);                                     /* e.g. 海水中 -> 水 */
        cnl_l_push(&R->today5elements, last);
        cnl_cpy(buf, "\xe5\xb1\x9e", 4); memcpy(buf + 3, last, 3); buf[6] = 0;
        cnl_l_push(&R->today5elements, buf);
    }
    cnl_l_push(&R->today5elements, "\xe5\xbb\xbf\xe5\x85\xab\xe5\xae\xbf");                   /* 廿八宿 */
    { char c0[4]; cnl_utf8_take(c0, R->today28star, 1); cnl_l_push(&R->today5elements, c0); }
    cnl_l_push(&R->today5elements, "\xe5\xae\xbf");                                           /* 宿 */
    cnl_l_push(&R->today5elements, "\xe5\x8d\x81\xe4\xba\x8c\xe7\xa5\x9e");                  /* 十二神 */
    cnl_l_push(&R->today5elements, R->today12day_officer);
    cnl_l_push(&R->today5elements, "\xe6\x97\xa5");                                           /* 日 */
}

static void cnl_9flystar(cnlunar_result *R) {
    int apart = cnl_daynum(R->year, R->month, R->day) - cnl_daynum(2019, 1, 17);
    static const int start[9] = {7, 3, 5, 6, 8, 1, 2, 4, 9};
    for (int i = 0; i < 9; ++i) {
        int v = cnl_mod(start[i] - 1 - apart, 9) + 1;
        R->the9flystar[i] = (char)('0' + v);
    }
    R->the9flystar[9] = 0;
}

static void cnl_lucky_gods_direction(cnlunar_result *R) {
    const char *names[5] = {"\xe5\x96\x9c\xe7\xa5\x9e", "\xe8\xb4\xa2\xe7\xa5\x9e", "\xe7\xa6\x8f\xe7\xa5\x9e", "\xe9\x98\xb3\xe8\xb4\xb5", "\xe9\x98\xb4\xe8\xb4\xb5"}; /* 喜神 财神 福神 阳贵 阴贵 */
    const char *tables[5] = { cnl_lucky_dir, cnl_wealth_dir, cnl_mascot_dir, cnl_sunnoble_dir, cnl_moonnoble_dir };
    int n = R->day_heaven_num;
    cnl_l_reset(&R->lucky_gods_direction);
    for (int i = 0; i < 5; ++i) {
        char tri[4];
        memcpy(tri, tables[i] + 3 * n, 3);
        tri[3] = 0;
        int dir = -1;
        for (int j = 0; j < 8; ++j) { char t2[4]; memcpy(t2, cnl_TRIGRAM_8 + 3 * j, 3); t2[3] = 0; if (strcmp(t2, tri) == 0) { dir = j; break; } }
        if (dir < 0) dir = 0;
        char buf[CNLUNAR_ITEM_MAX];
        cnl_cpy(buf, names[i], CNLUNAR_ITEM_MAX);
        { int _bl=(int)strlen(buf), _l=(int)strlen(cnl_DIRECTION[dir]); memcpy(buf+_bl, cnl_DIRECTION[dir], _l); buf[_bl+_l]=0; }
        cnl_l_push(&R->lucky_gods_direction, buf);
    }
}

static void cnl_fetal_god(cnlunar_result *R) {
    cnl_cpy(R->fetal_god, cnl_FETAL_GOD[R->day_heavenly_earth_num], 24);
}

static void cnl_twohour_lucky(cnlunar_result *R) {
    cnl_l_reset(&R->twohour_lucky);
    int today = R->day_heavenly_earth_num;
    int tomorrow = (today + 1) % 60;
    for (int t = 0; t < 2; ++t) {
        unsigned short v = cnl_TWOHOUR_LUCKY[t ? tomorrow : today];
        for (int i = 1; i <= 12; ++i) {
            int bit = 1 << (12 - i);
            cnl_l_push(&R->twohour_lucky, (v & bit) ? "\xe5\x87\xb6" : "\xe5\x90\x89"); /* 凶 / 吉 */
        }
    }
    if (R->twohour_lucky.n > 13) R->twohour_lucky.n = 13;
}

static void cnl_meridians(cnlunar_result *R) {
    int i = R->twohour_num % 12;
    cnl_cpy(R->meridians, cnl_MERIDIANS[i], 8);
    cnl_cpy(R->meridian_note, cnl_MERIDIAN_NOTE[i], CNLUNAR_STR_MAX);
    cnl_cpy(R->meridian_yi, cnl_MERIDIAN_YI[i], 64);
    cnl_cpy(R->meridian_ji, cnl_MERIDIAN_JI[i], 64);
}

/* ------------------ 节日 ------------------ */
static void cnl_holidays(cnlunar_result *R, cnl_scratch *cs) {
    /* 法定假日 */
    {
        char tmp[64];
        tmp[0] = 0;
        /* 清明节 (第 5 个节气) */
        if (strcmp(R->today_solar_terms, cnl_SOLAR_TERMS_NAME[6]) == 0) {  /* 清明 */
            strncat(tmp, "\xe6\xb8\x85\xe6\x98\x8e\xe8\x8a\x82 ", 63 - (int)strlen(tmp)); /* 清明节 */
        }
        {  /* solar legal holidays */
            static const int lm[3] = {1, 5, 10};
            static const char *const nm[3] = {"\xe5\x85\x83\xe6\x97\xa6\xe8\x8a\x82", "\xe5\x9b\xbd\xe9\x99\x85\xe5\x8a\xb3\xe5\x8a\xa8\xe8\x8a\x82", "\xe5\x9b\xbd\xe5\xba\x86\xe8\x8a\x82"};
            for (int i = 0; i < 3; ++i)
                if (R->month == lm[i] && R->day == 1) { strncat(tmp, nm[i], 63 - (int)strlen(tmp)); strncat(tmp, " ", 63 - (int)strlen(tmp)); }
        }
        if (!(R->lunar_month > 12)) {
            static const int lm2[3] = {1, 5, 8};
            static const int ld2[3] = {1, 5, 15};
            static const char *const nm2[3] = {"\xe6\x98\xa5\xe8\x8a\x82", "\xe7\xab\xaf\xe5\x8d\x88\xe8\x8a\x82", "\xe4\xb8\xad\xe7\xa7\x8b\xe8\x8a\x82"};
            for (int i = 0; i < 3; ++i)
                if (R->lunar_month == lm2[i] && R->lunar_day == ld2[i]) { strncat(tmp, nm2[i], 63 - (int)strlen(tmp)); break; }
        }
        /* strip + replace ' ' with ',' */
        {
            int len = (int)strlen(tmp);
            while (len > 0 && (tmp[len - 1] == ' ')) tmp[--len] = 0;
            char out[64]; int w = 0;
            for (int i = 0; i < len; ++i) out[w++] = (tmp[i] == ' ') ? ',' : tmp[i];
            out[w] = 0;
            cnl_cpy(R->holidays_legal, out, 64);
        }
    }
    /* 其他阳历假日 */
    R->holidays_other[0] = 0;
    {
        CNL_WS_LIST(cs, tmpL);
        int y = R->year, m = R->month, d = R->day;
        {
            int iw_y, iw_w, iw_d;
            cnl_iso(cnl_daynum(y, m, d), &iw_y, &iw_w, &iw_d);
            int t1_y, t1_w, t1_d;
            cnl_iso(cnl_daynum(y, m, 1), &t1_y, &t1_w, &t1_d);
            (void)t1_y; (void)t1_d;
            int wnum = iw_w - t1_w + 1;
            if (m == 5 && wnum == 2 && iw_d == 7) cnl_l_push(tmpL, "\xe6\xaf\x8d\xe4\xba\xb2\xe8\x8a\x82"); /* 母亲节 */
            if (m == 6 && wnum == 3 && iw_d == 7) cnl_l_push(tmpL, "\xe7\x88\xb6\xe4\xba\xb2\xe8\x8a\x82"); /* 父亲节 */
        }
        for (int i = cnl_OTHER_HOLIDAY_IDX[m - 1]; i < cnl_OTHER_HOLIDAY_IDX[m]; ++i)
            if (cnl_OTHER_HOLIDAY[i].day == R->day) cnl_l_push(tmpL, cnl_OTHER_HOLIDAY[i].name);
        if (tmpL->n > 0) {
            char out[80]; int w = 0;
            for (int i = 0; i < tmpL->n; ++i) {
                if (i) out[w++] = ',';
                int l = (int)strlen(tmpL->it[i]);
                for (int j = 0; j < l && w < 79; ++j) out[w++] = tmpL->it[i][j];
            }
            out[w] = 0;
            cnl_cpy(R->holidays_other, out, 80);
        }
    }
    /* 其他农历假日 */
    {
        char out[48];
        int idx = cnl_mod(R->lunar_month - 1, 12);
        out[0] = 0;
        for (int i = cnl_OTHER_LUNAR_HOLIDAY_IDX[idx]; i < cnl_OTHER_LUNAR_HOLIDAY_IDX[idx + 1]; ++i) {
            if (cnl_OTHER_LUNAR_HOLIDAY[i].day == R->lunar_day) {
                int l = (int)strlen(cnl_OTHER_LUNAR_HOLIDAY[i].name);
                for (int j = 0; j < l && j < 47; ++j) out[j] = cnl_OTHER_LUNAR_HOLIDAY[i].name[j];
                out[l] = 0;
                break;
            }
        }
        cnl_cpy(R->holidays_lunar, out, 48);
    }
}

/* ------------------ 彭祖百忌 ------------------ */
static void cnl_peng_taboo(cnlunar_result *R) {
    char a[32], b[32];
    cnl_utf8_take(a, cnl_PENG_TABOO[R->day_heaven_num], 9);
    cnl_utf8_take(b, cnl_PENG_TABOO[R->day_earth_num + 10], 9);
    int w = 0;
    for (int i = 0; a[i] && w < 63; ++i) R->peng_taboo[w++] = a[i];
    if (w < 63) R->peng_taboo[w++] = ',';
    for (int i = 0; b[i] && w < 63; ++i) R->peng_taboo[w++] = b[i];
    R->peng_taboo[w] = 0;
}


/* ====================== 神煞 matching + 宜忌 ====================== */
typedef struct {
    cnlunar_result *R;
    int men, sn, yhn, yen4;
    int tmd_m, tmd_d;
    int t4j[4][2], t4l[4][2];
    int tuidi_m, tuidi_d, tuidi_days;
} cnl_god_ctx;

/* byte offset of the arg-th (3-byte) char */
static int cnl_god_argval(const cnl_god_row *E, cnl_god_ctx *C) {
    switch (E->arg) {
    case 0: return C->men;
    case 1: return C->sn;
    case 2: return C->yhn;
    case 3: return C->yen4;
    case 4: return C->men % 6;
    }
    return 0;
}

static int cnl_row_match(const cnl_god_row *E, cnl_god_ctx *C) {
    const cnlunar_result *R = C->R;
    const char *d = R->day8char;
    switch (E->op) {
    case 1: { /* cin */
        int k = cnl_god_argval(E, C);
        if (3 * k + 3 > (int)strlen(E->base)) return 0;
        return cnl_bytes_in(d, 6, E->base + 3 * k, 3);
    }
    case 2: { /* cstem */
        int k = cnl_god_argval(E, C);
        if (3 * k + 3 > (int)strlen(E->base)) return 0;
        return memcmp(E->base + 3 * k, d, 3) == 0;
    }
    case 3: { /* cbr */
        int k = cnl_god_argval(E, C);
        if (3 * k + 3 > (int)strlen(E->base)) return 0;
        return memcmp(E->base + 3 * k, d + 3, 3) == 0;
    }
    case 4: { /* star0 */
        int k = cnl_god_argval(E, C);
        return memcmp(E->base + 3 * k, R->today28star, 3) == 0;
    }
    case 5: { /* dstemgrp */
        const char *grp = E->list[cnl_god_argval(E, C)];
        return cnl_bytes_in(grp, (int)strlen(grp), d, 3);
    }
    case 6: { /* dbrgrp */
        const char *grp = E->list[cnl_god_argval(E, C)];
        return cnl_bytes_in(grp, (int)strlen(grp), d + 3, 3);
    }
    case 7: /* dbrstr */
        return cnl_bytes_in(E->base, (int)strlen(E->base), d + 3, 3);
    case 8: { /* dlist */
        for (int i = 0; i < E->list_n; ++i)
            if (strcmp(E->list[i], d) == 0) return 1;
        return 0;
    }
    case 9: { /* dlistgrp: entry is one 干支 (6B) or a pair of them (12B) */
        int k = cnl_god_argval(E, C);
        if (k < 0 || k >= E->list_n) return 0;
        int len = (int)strlen(E->list[k]);
        if (len == 6) return strcmp(E->list[k], d) == 0;
        if (len == 12) return memcmp(E->list[k], d, 6) == 0 || memcmp(E->list[k] + 6, d, 6) == 0;
        return 0;
    }
    case 10: /* ldn */
        for (int i = 0; i < E->pairs_n; ++i) if (E->pairs[i] == R->lunar_day) return 1;
        return 0;
    case 11: { /* tmd — 四绝 uses 立X (t4j), 四离 uses 二分二至 (t4l) */
        const int (*tw)[2] = (strcmp(E->name, "四绝") == 0) ? (const int (*)[2])C->t4j : (const int (*)[2])C->t4l;
        for (int i = 0; i < 4; ++i)
            if (tw[i][0] == C->tmd_m && tw[i][1] == C->tmd_d) return 1;
        return 0;
    }
    case 12: /* lmnld */
        for (int i = 0; i < E->pairs_n; ++i)
            if (E->pairs[2 * i] == R->lunar_month && E->pairs[2 * i + 1] == R->lunar_day) return 1;
        return 0;
    case 13: /* lmnch */
        for (int i = 0; i < E->pairs_n; ++i)
            if (E->pairs[i] == R->lunar_month && strcmp(E->pstr[i], d) == 0) return 1;
        return 0;
    case 14: { /* dslice */
        int k = cnl_god_argval(E, C);
        if (6 * k + 6 > (int)strlen(E->base)) return 0;
        return memcmp(E->base + 6 * k, d, 6) == 0;
    }
    case 15: { /* dbrslice */
        int k = cnl_god_argval(E, C);
        if (9 * k + 9 > (int)strlen(E->base)) return 0;
        return cnl_bytes_in(E->base + 9 * k, 9, d + 3, 3);
    }
    case 16: /* tuidi */
        return C->tuidi_days >= 0 && C->tuidi_days < 18;
    case 17: /* b3he */
        return ((R->day_earth_num - C->men) % 4 == 0);
    case 18: /* bsuip */
        return (R->day_earth_num == (R->year_earth_num + 6) % 12);
    case 19: /* btian */
        return (R->day_heavenly_earth_num % 15 < 5 && R->day_heavenly_earth_num / 15 != 2);
    case 20: { /* tiande */
        const char *judge = (R->lunar_season_type == 0) ? d + 3 : d;
        const char *grp = E->list[C->men];
        return cnl_bytes_in(grp, (int)strlen(grp), judge, 3);
    }
    case 21: { /* bujiang */
        const char *grp = cnl_BUJIANG[C->men];
        for (int k = 0; k < 13 && 6 * k + 6 <= (int)strlen(grp); ++k)
            if (memcmp(grp + 6 * k, d, 6) == 0) return 1;
        return 0;
    }
    }
    return 0;
}

/* deIsBadThing from the first six angel rows */
static void cnl_de_is_bad_thing(cnl_god_ctx *C, cnlunar_list *out) {
    cnl_l_reset(out);
    for (int i = 0; i < 6; ++i) {
        const cnl_god_row *E = &cnl_ANGEL_ROWS[i];
        if (cnl_l_has(&C->R->good_god_name, E->name))
            cnl_l_union_arr(out, E->bad, E->n_bad);
    }
}

/* main 神煞 computation; fills good_god_name/bad_god_name/good_thing/bad_thing
   and levels. Mirrors get_AngelDemon() + getTodayThingLevel(). */
static void cnl_angel_demon(cnlunar_result *R, cnl_scratch *cs) {
    cnl_god_ctx C;
    memset(&C, 0, sizeof(C));
    C.R = R;
    C.men = (R->options & CNLUNAR_GODTYPE_CNLUNAR) ? (R->lunar_month - 1 + 2) % 12 : R->month_earth_num;
    C.sn = R->lunar_season_num;
    C.yhn = R->year_heaven_num;
    C.yen4 = R->year_earth_num % 4;
    {
        int dn = cnl_daynum(R->year, R->month, R->day) + 1;
        int ty, tm, td;
        cnl_ymd(dn, &ty, &tm, &td);
        C.tmd_m = tm; C.tmd_d = td;
    }
    /* t4j 立春夏秋冬 ; t4l 春分夏至秋分冬至 (this year) */
    {
        static const int t4j_idx[4] = {2, 8, 14, 20};
        static const int t4l_idx[4] = {5, 11, 17, 23};
        for (int i = 0; i < 4; ++i) {
            C.t4j[i][0] = R->this_year_solar_terms[t4j_idx[i]].date.month;
            C.t4j[i][1] = R->this_year_solar_terms[t4j_idx[i]].date.day;
            C.t4l[i][0] = R->this_year_solar_terms[t4l_idx[i]].date.month;
            C.t4l[i][1] = R->this_year_solar_terms[t4l_idx[i]].date.day;
        }
        /* 土王用事 */
        int cnt = 0;
        for (int i = 0; i < 4; ++i) {
            int a_mon = C.t4j[i][0], a_day = C.t4j[i][1];
            if (a_mon < C.tmd_m || (a_mon == C.tmd_m && a_day < C.tmd_d)) ++cnt;
        }
        int k = cnt % 4;
        C.tuidi_m = C.t4j[k][0]; C.tuidi_d = C.t4j[k][1];
        /* Python: (datetime(target) - datetime(date,h,m)).days — floor of the
           day-count including the hour-of-day remainder */
        {
            long long tsec = (long long)(cnl_daynum(R->next_solar_term_year, C.tuidi_m, C.tuidi_d) - cnl_daynum(R->year, R->month, R->day)) * 86400LL
                           - (long long)(R->hour * 3600 + R->minute * 60);
            long long d = tsec / 86400LL;
            if (tsec < 0 && tsec % 86400LL != 0) d -= 1;   /* floor division */
            C.tuidi_days = (int)d;
        }
    }
    const char *d = R->day8char;
    int isyd = (R->options & CNLUNAR_YEARGOD_NODUTY) ? 0 : 1;   /* isYeargodDuty */

    /* officer 宜忌 + day8 char things + extra rules */
    {
        const cnl_officer_t *ot = 0;
        for (int i = 0; i < 12; ++i)
            if (strcmp(cnl_officer_things[i].ch, R->today12day_officer) == 0) { ot = &cnl_officer_things[i]; break; }
        if (ot) {
            cnl_l_union_arr(&R->good_thing, ot->good, ot->n_good);
            cnl_l_union_arr(&R->bad_thing, ot->bad, ot->n_bad);
        }
    }
    for (int i = 0; i < 16; ++i) {
        const cnl_d8thing_t *t = &cnl_day8_char_thing[i];
        if (strstr(d, t->ch)) {
            cnl_l_union_arr(&R->good_thing, t->good, t->n_good);
            cnl_l_union_arr(&R->bad_thing, t->bad, t->n_bad);
        }
    }
    /* 节气间差类 (取鱼/畋猎/伐木) */
    {
        int nsn = R->next_solar_num;
        const char *o = R->today12day_officer;
        int is_zq_r = (strcmp(o, "\xe6\x89\xa7") == 0 || strcmp(o, "\xe5\x8d\xb1") == 0 || strcmp(o, "\xe6\x94\xb6") == 0); /* 执 危 收 */
        if (nsn >= 4 && nsn <= 8 && is_zq_r)
            cnl_l_add(&R->good_thing, "\xe5\x8f\x96\xe9\xb1\xbc");                       /* 取鱼 */
        if ((nsn >= 20 && nsn < 24) || (nsn >= 0 && nsn < 3))
            if (is_zq_r)
                cnl_l_add(&R->good_thing, "\xe7\x95\x8b\xe7\x8c\x8e");                  /* 畋猎 */
        if ((nsn >= 21 && nsn < 24) || (nsn >= 0 && nsn < 3)) {
            /* Python: (o in ['危'] or d in ['午','申']) — the latter never true as d is 2 chars */
            if (strcmp(o, "\xe5\x8d\xb1") == 0) cnl_l_add(&R->good_thing, "\xe4\xbc\x90\xe6\x9c\xa8"); /* 伐木 */
        }
    }
    {
        static const int a1[6] = {1, 6, 15, 19, 21, 23};
        for (int i = 0; i < 6; ++i) if (R->lunar_day == a1[i]) cnl_l_add(&R->bad_thing, "\xe6\x95\xb4\xe6\x89\x8b\xe8\xb6\xb3\xe7\x94\xb2"); /* 整手足甲 */
        if (R->lunar_day == 12 || R->lunar_day == 15) {
            cnl_l_add(&R->bad_thing, "\xe6\x95\xb4\xe5\xae\xb9");                          /* 整容 */
            cnl_l_add(&R->bad_thing, "\xe5\x89\x83\xe5\xa4\xb4");                          /* 剃头 */
        }
        if (R->lunar_day == 15 || R->phase_of_moon[0] != 0)
            cnl_l_add(&R->bad_thing, "\xe6\xb1\x82\xe5\x8c\xbb\xe7\x96\x97\xe7\x97\x85"); /* 求医疗病 */
    }

    /* angel pass */
    for (int i = 0; i < 71; ++i) {
        const cnl_god_row *E = &cnl_ANGEL_ROWS[i];
        if (!isyd && strstr(E->name, "\xe5\xb2\x81") != NULL) continue; /* 岁 */
        if (cnl_row_match(E, &C)) {
            cnl_l_add(&R->good_god_name, E->name);
            cnl_l_union_arr(&R->good_thing, E->good, E->n_good);
            cnl_l_union_arr(&R->bad_thing, E->bad, E->n_bad);
        }
    }
    cnl_l_dedup(&R->good_thing);
    cnl_l_dedup(&R->bad_thing);
    /* demon pass */
    for (int i = 0; i < 100; ++i) {
        const cnl_god_row *E = &cnl_DEMON_ROWS[i];
        if (!isyd && strstr(E->name, "\xe5\xb2\x81") != NULL) continue;
        if (cnl_row_match(E, &C)) {
            cnl_l_add(&R->bad_god_name, E->name);
            cnl_l_union_arr(&R->good_thing, E->good, E->n_good);
            cnl_l_union_arr(&R->bad_thing, E->bad, E->n_bad);
        }
    }
    cnl_l_dedup(&R->good_thing);
    cnl_l_dedup(&R->bad_thing);

    /* ============ 宜忌等第表 ============ */
    int l = -1;
    {
        /* todayAllGodName = good + bad + officer日 */
        CNL_WS_LIST(cs, allgods);
        for (int i = 0; i < R->good_god_name.n; ++i) cnl_l_push(allgods, R->good_god_name.it[i]);
        for (int i = 0; i < R->bad_god_name.n; ++i) cnl_l_push(allgods, R->bad_god_name.it[i]);
        {
            char oday[8];
            cnl_cpy(oday, R->today12day_officer, 4);
            strncat(oday, "\xe6\x97\xa5", 7);  /* 日 */
            cnl_l_push(allgods, oday);
        }
        char mb[4];
        cnl_utf8_take(mb, R->month8char, 1); /* month8char 只有一个字符是地支? no: month8Char[1] = 地支 */
        (void)mb;
        const char *mbranch = R->month8char + 3;             /* month8Char[1] = 地支字符 */
        for (int bi = 0; bi < allgods->n; ++bi) {
            const char *gname = allgods->it[bi];
            const cnl_tl_entry *found = 0;
            for (int ti = 0; ti < 11; ++ti)
                if (strcmp(cnl_thing_level[ti].key, gname) == 0) { found = &cnl_thing_level[ti]; break; }
            if (!found) continue;
            for (int si = 0; si < found->n; ++si) {
                const cnl_tl_sub *sub = &found->sub[si];
                if (!cnl_bytes_in(sub->pat, (int)strlen(sub->pat), mbranch, 3)) continue;
                int hit = 0;
                for (int gi = 0; gi < sub->n; ++gi) {
                    if (cnl_l_has(allgods, sub->gods[gi]) && sub->level > l) { l = sub->level; hit = 1; break; }
                }
                (void)hit;
            }
        }
    }
    R->today_level = l;
    {
        static const char *const level_names[7] = {
            "\xe4\xb8\x8a\xef\xbc\x9a\xe5\x90\x89\xe8\xb6\xb3\xe8\x83\x9c\xe5\x87\xb6\xef\xbc\x8c\xe4\xbb\x8e\xe5\xae\x9c\xe4\xb8\x8d\xe4\xbb\x8e\xe5\xbf\x8c\xe3\x80\x82",
            "\xe4\xb8\x8a\xe6\xac\xa1\xef\xbc\x9a\xe5\x90\x89\xe8\xb6\xb3\xe6\x8a\xb5\xe5\x87\xb6\xef\xbc\x8c\xe9\x81\x87\xe5\xbe\xb7\xe4\xbb\x8e\xe5\xae\x9c\xe4\xb8\x8d\xe4\xbb\x8e\xe5\xbf\x8c\xef\xbc\x8c\xe4\xb8\x8d\xe9\x81\x87\xe4\xbb\x8e\xe5\xae\x9c\xe4\xba\xa6\xe4\xbb\x8e\xe5\xbf\x8c\xe3\x80\x82",
            "\xe4\xb8\xad\xef\xbc\x9a\xe5\x90\x89\xe4\xb8\x8d\xe6\x8a\xb5\xe5\x87\xb6\xef\xbc\x8c\xe9\x81\x87\xe5\xbe\xb7\xe4\xbb\x8e\xe5\xae\x9c\xe4\xb8\x8d\xe4\xbb\x8e\xe5\xbf\x8c\xef\xbc\x8c\xe4\xb8\x8d\xe9\x81\x87\xe4\xbb\x8e\xe5\xbf\x8c\xe4\xb8\x8d\xe4\xbb\x8e\xe5\xae\x9c\xe3\x80\x82",
            "\xe4\xb8\xad\xe6\xac\xa1\xef\xbc\x9a\xe5\x87\xb6\xe8\x83\x9c\xe4\xba\x8e\xe5\x90\x89\xef\xbc\x8c\xe9\x81\x87\xe5\xbe\xb7\xe4\xbb\x8e\xe5\xae\x9c\xe4\xba\xa6\xe4\xbb\x8e\xe5\xbf\x8c\xef\xbc\x8c\xe4\xb8\x8d\xe9\x81\x87\xe4\xbb\x8e\xe5\xbf\x8c\xe4\xb8\x8d\xe4\xbb\x8e\xe5\xae\x9c\xe3\x80\x82",
            "\xe4\xb8\x8b\x3a\xe5\x87\xb6\xe5\x8f\x88\xe9\x80\xa2\xe5\x87\xb6\xef\xbc\x8c\xe9\x81\x87\xe5\xbe\xb7\xe4\xbb\x8e\xe5\xbf\x8c\xe4\xb8\x8d\xe4\xbb\x8e\xe5\xae\x9c\xef\xbc\x8c\xe4\xb8\x8d\xe9\x81\x87\xe8\xaf\xb8\xe4\xba\x8b\xe7\x9a\x86\xe5\xbf\x8c\xe3\x80\x82",
            "\xe4\xb8\x8b\xe4\xb8\x8b\xef\xbc\x9a\xe5\x87\xb6\xe5\x8f\xa0\xe5\xa4\xa7\xe5\x87\xb6\xef\xbc\x8c\xe9\x81\x87\xe5\xbe\xb7\xe4\xba\xa6\xe8\xaf\xb8\xe4\xba\x8b\xe7\x9a\x86\xe5\xbf\x8c\xe3\x80\x82\xef\xbc\x88\xe5\x8d\xaf\xe9\x85\x89\xe6\x9c\x88\xef\xbc\x8c\xe7\x81\xbe\xe7\x85\x9e\xe9\x81\x87\xe6\x9c\x88\xe7\xa0\xb4\xe3\x80\x81\xe6\x9c\x88\xe5\x8e\x8c\xef\xbc\x8c\xe6\x9c\x88\xe5\x8e\x8c\xe9\x81\x87\xe7\x81\xbe\xe7\x85\x9e\xe3\x80\x81\xe6\x9c\x88\xe7\xa0\xb4\xef\xbc\x89",
            "\xe6\x97\xa0"
        };
        if (l == -1) cnl_cpy(R->today_level_name, "\xe6\x97\xa0", 4); /* 无 */
        else cnl_cpy(R->today_level_name, level_names[l], CNLUNAR_STR_MAX);
    }
    int thing_level;
    {
        int is_de = 0;
        for (int i = 0; i < R->good_god_name.n; ++i) {
            const char *g = R->good_god_name.it[i];
            if (strcmp(g, "岁德") == 0 || strcmp(g, "岁德合") == 0 || strcmp(g, "月德") == 0 ||
                strcmp(g, "月德合") == 0 || strcmp(g, "天德") == 0 || strcmp(g, "天德合") == 0) { is_de = 1; break; }
        }
        R->is_de = is_de;
        if (l == 5) thing_level = 3;
        else if (l == 4) thing_level = is_de ? 2 : 3;
        else if (l == 3) thing_level = is_de ? 1 : 2;
        else if (l == 2) thing_level = is_de ? 0 : 2;
        else if (l == 1) thing_level = is_de ? 0 : 1;
        else if (l == 0) thing_level = 0;
        else thing_level = 1;
        static const char *const tl_names[4] = {
            "\xe4\xbb\x8e\xe5\xae\x9c\xe4\xb8\x8d\xe4\xbb\x8e\xe5\xbf\x8c",        /* 从宜不从忌 */
            "\xe4\xbb\x8e\xe5\xae\x9c\xe4\xba\xa6\xe4\xbb\x8e\xe5\xbf\x8c",        /* 从宜亦从忌 */
            "\xe4\xbb\x8e\xe5\xbf\x8c\xe4\xb8\x8d\xe4\xbb\x8e\xe5\xae\x9c",        /* 从忌不从宜 */
            "\xe8\xaf\xb8\xe4\xba\x8b\xe7\x9a\x86\xe5\xbf\x8c"                      /* 诸事皆忌 */
        };
        cnl_cpy(R->thing_level_name, tl_names[thing_level], 16);
        R->thing_level = thing_level;
    }

    /* deIsBadThing */
    CNL_WS_LIST(cs, dibt);
    cnl_de_is_bad_thing(&C, dibt);

    /* 今日凶吉判断: 0 从宜不从忌 / 1 从宜亦从忌 / 2 从忌不从宜 / 3 诸事皆忌 */
    if (thing_level == 3) {
        cnl_l_reset(&R->good_thing); cnl_l_add(&R->good_thing, "\xe8\xaf\xb8\xe4\xba\x8b\xe4\xb8\x8d\xe5\xae\x9c");
        cnl_l_reset(&R->bad_thing);  cnl_l_add(&R->bad_thing, "\xe8\xaf\xb8\xe4\xba\x8b\xe4\xb8\x8d\xe5\xae\x9c");
    }
    else if (thing_level == 2) { /* 从忌不从宜: good 中与 bad 相同者移除 */
        CNL_WS_LIST(cs, rm2);
        for (int i = 0; i < R->bad_thing.n; ++i) if (cnl_l_has(&R->good_thing, R->bad_thing.it[i])) cnl_l_add(rm2, R->bad_thing.it[i]);
        cnl_l_rm_list(&R->good_thing, rm2);
    }
    else if (thing_level == 1) { /* 从宜亦从忌: 交集从两表移除 */
        CNL_WS_LIST(cs, rm1);
        for (int i = 0; i < R->good_thing.n; ++i) if (cnl_l_has(&R->bad_thing, R->good_thing.it[i])) cnl_l_add(rm1, R->good_thing.it[i]);
        cnl_l_rm_list(&R->good_thing, rm1);
        cnl_l_rm_list(&R->bad_thing, rm1);
    }
    else { /* 从宜不从忌: bad 中与 good 相同者移除 */
        CNL_WS_LIST(cs, rm0);
        for (int i = 0; i < R->bad_thing.n; ++i) if (cnl_l_has(&R->good_thing, R->bad_thing.it[i])) cnl_l_add(rm0, R->bad_thing.it[i]);
        cnl_l_rm_list(&R->bad_thing, rm0);
    }

    if (thing_level != 3) {
        if (cnl_l_has(&R->good_thing, "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b") && cnl_l_has(&R->good_thing, "\xe5\xb8\x83\xe6\x94\xbf\xe4\xba\x8b"))
            cnl_l_rm(&R->good_thing, "\xe5\xb8\x83\xe6\x94\xbf\xe4\xba\x8b");
        {
            int isde_she = 0;
            for (int i = 0; i < R->good_god_name.n; ++i) {
                const char *g = R->good_god_name.it[i];
                if (strcmp(g, "\xe6\x9c\x88\xe5\xbe\xb7\xe5\x90\x88") == 0 || strcmp(g, "\xe5\xa4\xa9\xe5\xbe\xb7\xe5\x90\x88") == 0 ||
                    strcmp(g, "\xe5\xa4\xa9\xe8\xb5\xa6") == 0 || strcmp(g, "\xe5\xa4\xa9\xe6\x84\xbf") == 0 ||
                    strcmp(g, "\xe6\x9c\x88\xe6\x81\xa9") == 0 || strcmp(g, "\xe5\x9b\x9b\xe7\x9b\xb8") == 0 ||
                    strcmp(g, "\xe6\x97\xb6\xe5\xbe\xb7") == 0 || (isyd && strcmp(g, "\xe5\xb2\x81\xe5\xbe\xb7\xe5\x90\x88") == 0)) {
                    isde_she = 1; break;
                }
            }
            if (isde_she && thing_level != 2) {
                

                static const char *const rm9[] = {"\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3","\xe5\xae\x89\xe5\xba\x8a","\xe7\xbb\x8f\xe7\xbb\x9c","\xe9\x85\x9d\xe9\x85\xbf","\xe5\xbc\x80\xe5\xb8\x82","\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93","\xe7\xba\xb3\xe8\xb4\xa2","\xe5\xbc\x80\xe4\xbb\x93\xe5\xba\x93","\xe5\x87\xba\xe8\xb4\xa7\xe8\xb4\xa2"};
                cnl_l_rm_arr(&R->bad_thing, rm9, 9);
                cnl_l_union_from(&R->bad_thing, dibt);
            }
        }
        /* 天狗/寅: 祭祀 */
        if (cnl_l_has(&R->bad_god_name, "\xe5\xa4\xa9\xe7\x8b\x97") || cnl_bytes_in(d, 6, "\xe5\xaf\x85", 3)) {
            cnl_l_add(&R->bad_thing, "\xe7\xa5\xad\xe7\xa5\x80");
            cnl_l_rm(&R->good_thing, "\xe7\xa5\xad\xe7\xa5\x80");
            cnl_l_rm(&R->good_thing, "\xe6\xb1\x82\xe7\xa6\x8f");
            cnl_l_rm(&R->good_thing, "\xe7\xa5\x88\xe5\x97\xa3");
        }
        /* 卯日 忌穿井 */
        if (cnl_bytes_in(d, 6, "\xe5\x8d\xaf", 3)) {
            cnl_l_add(&R->bad_thing, "\xe7\xa9\xbf\xe4\xba\x95");
            cnl_l_rm(&R->good_thing, "\xe7\xa9\xbf\xe4\xba\x95");
            cnl_l_rm(&R->good_thing, "\xe5\xbc\x80\xe6\xb8\xa0");
        }
        /* 壬日 忌开渠 */
        if (cnl_bytes_in(d, 6, "\xe5\xa3\xac", 3)) {
            cnl_l_add(&R->bad_thing, "\xe5\xbc\x80\xe6\xb8\xa0");
            cnl_l_rm(&R->good_thing, "\xe5\xbc\x80\xe6\xb8\xa0");
            cnl_l_rm(&R->good_thing, "\xe7\xa9\xbf\xe4\xba\x95");
        }
        /* 巳日 忌出行 */
        if (cnl_bytes_in(d, 6, "\xe5\xb7\xb3", 3)) {
            cnl_l_add(&R->bad_thing, "\xe5\x87\xba\xe8\xa1\x8c");
            cnl_l_rm(&R->good_thing, "\xe5\x87\xba\xe8\xa1\x8c");
            cnl_l_rm(&R->good_thing, "\xe5\x87\xba\xe5\xb8\x88");
            cnl_l_rm(&R->good_thing, "\xe9\x81\xa3\xe4\xbd\xbf");
        }
        /* 酉日 忌宴会 */
        if (cnl_bytes_in(d, 6, "\xe9\x85\x89", 3)) {
            cnl_l_add(&R->bad_thing, "\xe5\xae\xb4\xe4\xbc\x9a");
            cnl_l_rm(&R->good_thing, "\xe5\xae\xb4\xe4\xbc\x9a");
            cnl_l_rm(&R->good_thing, "\xe5\xba\x86\xe8\xb5\x90");
            cnl_l_rm(&R->good_thing, "\xe8\xb5\x8f\xe8\xb4\xba");
        }
        /* 丁日 忌剃头 */
        if (cnl_bytes_in(d, 6, "\xe4\xb8\x81", 3)) {
            cnl_l_add(&R->bad_thing, "\xe5\x89\x83\xe5\xa4\xb4");
            cnl_l_rm(&R->good_thing, "\xe5\x89\x83\xe5\xa4\xb4");
            cnl_l_rm(&R->good_thing, "\xe6\x95\xb4\xe5\xae\xb9");
        }
        if (R->today_level == 0 && thing_level == 0)
            cnl_l_union_from(&R->bad_thing, dibt);
        if (R->today_level == 1) {
            cnl_l_union_from(&R->bad_thing, dibt);
            if (!cnl_l_has(&R->bad_thing, "\xe7\xa5\x88\xe7\xa6\x8f"))
                cnl_l_rm(&R->bad_thing, "\xe6\xb1\x82\xe5\x97\xa3");
            if (!cnl_l_has(&R->bad_thing, "\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb") && !R->is_de) {
                static const char *const rm4[] = {"\xe5\x86\xa0\xe5\xb8\xa6","\xe7\xba\xb3\xe9\x87\x87\xe9\x97\xae\xe5\x90\x8d","\xe5\xab\x81\xe5\xa8\xb6","\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3"};
                cnl_l_rm_arr(&R->bad_thing, rm4, 4);
            }
            if (!cnl_l_has(&R->bad_thing, "\xe5\xab\x81\xe5\xa8\xb6") && !R->is_de) {
                if (!cnl_l_has(&R->good_god_name, "\xe4\xb8\x8d\xe5\xb0\x86")) {
                    static const char *const rm6[] = {"\xe5\x86\xa0\xe5\xb8\xa6","\xe7\xba\xb3\xe9\x87\x87\xe9\x97\xae\xe5\x90\x8d","\xe7\xbb\x93\xe5\xa9\x9a\xe5\xa7\xbb","\xe8\xbf\x9b\xe4\xba\xba\xe5\x8f\xa3","\xe6\x90\xac\xe7\xa7\xbb","\xe5\xae\x89\xe5\xba\x8a"};
                    cnl_l_rm_arr(&R->bad_thing, rm6, 6);
                }
            }
        }
        /* 亥日 忌嫁娶 */
        if (cnl_bytes_in(d, 6, "\xe4\xba\xa5", 3))
            cnl_l_add(&R->bad_thing, "\xe5\xab\x81\xe5\xa8\xb6");
        if (R->today_level == 1 && !R->is_de) {
            if (!cnl_l_has(&R->bad_thing, "\xe6\x90\xac\xe7\xa7\xbb"))
                cnl_l_rm(&R->bad_thing, "\xe5\xae\x89\xe5\xba\x8a");
            if (!cnl_l_has(&R->bad_thing, "\xe5\xae\x89\xe5\xba\x8a"))
                cnl_l_rm(&R->bad_thing, "\xe6\x90\xac\xe7\xa7\xbb");
            if (!cnl_l_has(&R->bad_thing, "\xe8\xa7\xa3\xe9\x99\xa4")) {
                static const char *const r3[] = {"\xe6\x95\xb4\xe5\xae\xb9","\xe5\x89\x83\xe5\xa4\xb4","\xe6\x95\xb4\xe6\x89\x8b\xe8\xb6\xb3\xe7\x94\xb2"};
                cnl_l_rm_arr(&R->bad_thing, r3, 3);
            }
            if (!cnl_l_has(&R->bad_thing, "\xe4\xbf\xae\xe9\x80\xa0") || !cnl_l_has(&R->bad_thing, "\xe7\xab\x96\xe6\x9f\xb1\xe4\xb8\x8a\xe6\xa2\x81")) {
                static const char *const r14[] = {"\xe4\xbf\xae\xe5\xae\xab\xe5\xae\xa4","\xe7\xbc\xae\xe5\x9f\x8e\xe9\x83\xad","\xe6\x95\xb4\xe6\x89\x8b\xe8\xb6\xb3\xe7\x94\xb2","\xe7\xad\x91\xe6\x8f\x90","\xe4\xbf\xae\xe4\xbb\x93\xe5\xba\x93","\xe9\xbc\x93\xe9\x93\xb8","\xe8\x8b\xab\xe7\x9b\x96","\xe4\xbf\xae\xe7\xbd\xae\xe4\xba\xa7\xe5\xae\xa4","\xe5\xbc\x80\xe6\xb8\xa0\xe7\xa9\xbf\xe4\xba\x95","\xe5\xae\x89\xe7\xa2\x93\xe7\xa1\x99","\xe8\xa1\xa5\xe5\x9e\xa3\xe5\xa1\x9e\xe7\xa9\xb4","\xe4\xbf\xae\xe9\xa5\xb0\xe5\x9e\xa3\xe5\xa2\x99","\xe5\xb9\xb3\xe6\xb2\xbb\xe9\x81\x93\xe6\xb6\x82","\xe7\xa0\xb4\xe5\xb1\x8b\xe5\x9d\x8f\xe5\x9e\xa3"};
                cnl_l_rm_arr(&R->bad_thing, r14, 14);
            }
        }
        if (R->today_level == 1) {
            if (!cnl_l_has(&R->bad_thing, "\xe5\xbc\x80\xe5\xb8\x82")) {
                static const char *const r4b[] = {"\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93","\xe7\xba\xb3\xe8\xb4\xa2","\xe5\xbc\x80\xe4\xbb\x93\xe5\xba\x93","\xe5\x87\xba\xe8\xb4\xa7\xe8\xb4\xa2"};
                cnl_l_rm_arr(&R->bad_thing, r4b, 4);
            }
            if (!cnl_l_has(&R->bad_thing, "\xe7\xba\xb3\xe8\xb4\xa2")) {
                static const char *const r2b[] = {"\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93","\xe5\xbc\x80\xe5\xb8\x82"};
                cnl_l_rm_arr(&R->bad_thing, r2b, 2);
            }
            if (!cnl_l_has(&R->bad_thing, "\xe7\xab\x8b\xe5\x88\xb8\xe4\xba\xa4\xe6\x98\x93")) {
                static const char *const r4c[] = {"\xe7\xba\xb3\xe8\xb4\xa2","\xe5\xbc\x80\xe5\xb8\x82","\xe5\xbc\x80\xe4\xbb\x93\xe5\xba\x93","\xe5\x87\xba\xe8\xb4\xa7\xe8\xb4\xa2"};
                cnl_l_rm_arr(&R->bad_thing, r4c, 4);
            }
            if (!cnl_l_has(&R->bad_thing, "\xe7\x89\xa7\xe5\x85\xbb"))
                cnl_l_rm(&R->bad_thing, "\xe7\xba\xb3\xe7\x95\x9c");
            if (!cnl_l_has(&R->bad_thing, "\xe7\xba\xb3\xe7\x95\x9c"))
                cnl_l_rm(&R->bad_thing, "\xe7\x89\xa7\xe5\x85\xbb");
            if (cnl_l_has(&R->good_thing, "\xe5\xae\x89\xe8\x91\xac"))
                cnl_l_rm(&R->bad_thing, "\xe5\x90\xaf\xe6\x94\x92");
            if (cnl_l_has(&R->good_thing, "\xe5\x90\xaf\xe6\x94\x92"))
                cnl_l_rm(&R->bad_thing, "\xe5\xae\x89\xe8\x91\xac");
        }
        if (cnl_l_has(&R->bad_thing, "\xe8\xaf\x8f\xe5\x91\xbd\xe5\x85\xac\xe5\x8d\xbf") || cnl_l_has(&R->bad_thing, "\xe6\x8b\x9b\xe8\xb4\xa4")) {
            cnl_l_rm(&R->good_thing, "\xe6\x96\xbd\xe6\x81\xa9");
            cnl_l_rm(&R->good_thing, "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4");
        }
        if (cnl_l_has(&R->bad_thing, "\xe6\x96\xbd\xe6\x81\xa9") || cnl_l_has(&R->bad_thing, "\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4")) {
            cnl_l_rm(&R->good_thing, "\xe8\xaf\x8f\xe5\x91\xbd\xe5\x85\xac\xe5\x8d\xbf");
            cnl_l_rm(&R->good_thing, "\xe6\x8b\x9b\xe8\xb4\xa4");
        }
        if (cnl_l_has(&R->good_thing, "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b") && cnl_l_has(&R->bad_god_name, "\xe5\xbe\x80\xe4\xba\xa1")) {
            cnl_l_rm(&R->good_thing, "\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b");
            cnl_l_add(&R->good_thing, "\xe5\xb8\x83\xe6\x94\xbf\xe4\xba\x8b");
        }
        if (cnl_l_has(&R->bad_god_name, "\xe6\x9c\x88\xe5\x8e\x8c")) {
            static const char *const r5g[] = {"\xe9\xa2\x81\xe8\xaf\x8f","\xe6\x96\xbd\xe6\x81\xa9","\xe6\x8b\x9b\xe8\xb4\xa4","\xe4\xb8\xbe\xe6\xad\xa3\xe7\x9b\xb4","\xe5\xae\xa3\xe6\x94\xbf\xe4\xba\x8b"};
            cnl_l_rm_arr(&R->good_thing, r5g, 5);
            cnl_l_add(&R->good_thing, "\xe5\xb8\x83\xe6\x94\xbf\xe4\xba\x8b");
            cnl_l_add(&R->bad_thing, "\xe8\xa1\xa5\xe5\x9e\xa3");
            if (cnl_l_has(&R->bad_god_name, "\xe5\x9c\x9f\xe5\xba\x9c") || cnl_l_has(&R->bad_god_name, "\xe5\x9c\x9f\xe7\xac\xa6") || cnl_l_has(&R->bad_god_name, "\xe5\x9c\xb0\xe5\x9b\x8a"))
                cnl_l_rm(&R->good_thing, "\xe5\xa1\x9e\xe7\xa9\xb4");
        }
        if (strstr(R->today12day_officer, "\xe5\xbc\x80")) {
            static const char *const r3 = "\x0";
            (void)r3;
            cnl_l_rm(&R->good_thing, "\xe7\xa0\xb4\xe5\x9c\x9f");
            cnl_l_rm(&R->good_thing, "\xe5\xae\x89\xe8\x91\xac");
            cnl_l_rm(&R->good_thing, "\xe5\x90\xaf\xe6\x94\x92");
        }
        if (cnl_l_has(&R->bad_god_name, "\xe5\x9b\x9b\xe5\xbf\x8c") || cnl_l_has(&R->bad_god_name, "\xe5\x9b\x9b\xe7\xa9\xb7")) {
            cnl_l_add(&R->bad_thing, "\xe5\xae\x89\xe8\x91\xac");
            cnl_l_rm(&R->good_thing, "\xe7\xa0\xb4\xe5\x9c\x9f");
            cnl_l_rm(&R->good_thing, "\xe5\x90\xaf\xe6\x94\x92");
        }
        if (cnl_l_has(&R->good_god_name, "\xe9\xb8\xa3\xe5\x90\xa0") || cnl_l_has(&R->good_god_name, "\xe9\xb8\xa3\xe5\x90\xa0\xe5\xaf\xb9")) {
            cnl_l_rm(&R->good_thing, "\xe7\xa0\xb4\xe5\x9c\x9f");
            cnl_l_rm(&R->good_thing, "\xe5\x90\xaf\xe6\x94\x92");
        }
        /* 德和与赦愿所汇 — ['空','甲戌','空','丙申','空','甲子','戊申','庚辰','辛卯','甲子','空','甲子'][lmn-1] in d */
        {
            static const char *const mday12[12] = {"空","甲戌","空","丙申","空","甲子","戊申","庚辰","辛卯","甲子","空","甲子"};
            const char *probe = mday12[(R->lunar_month - 1) % 12];
            if (strstr(d, probe))
                cnl_l_reset(&R->bad_thing), cnl_l_add(&R->bad_thing, "\xe8\xaf\xb8\xe4\xba\x8b\xe4\xb8\x8d\xe5\xbf\x8c");
        }
        if ((cnl_l_has(&R->good_god_name, "\xe5\xb2\x81\xe5\xbe\xb7\xe5\x90\x88") || cnl_l_has(&R->good_god_name, "\xe6\x9c\x88\xe5\xbe\xb7\xe5\x90\x88") || cnl_l_has(&R->good_god_name, "\xe5\xa4\xa9\xe5\xbe\xb7\xe5\x90\x88")) &&
            (cnl_l_has(&R->good_god_name, "\xe5\xa4\xa9\xe8\xb5\xa6") || cnl_l_has(&R->good_god_name, "\xe5\xa4\xa9\xe6\x84\xbf"))) {
            cnl_l_reset(&R->bad_thing);
            cnl_l_add(&R->bad_thing, "\xe8\xaf\xb8\xe4\xba\x8b\xe4\xb8\x8d\xe5\xbf\x8c");
        }
    }

    /* rmThing: bad ∩ good（两遍扫描，去掉 1KB 栈上 rm[] 索引数组） */
    {
        int rn = 0;
        const char *first = NULL;
        for (int i = 0; i < R->bad_thing.n; ++i)
            if (cnl_l_has(&R->good_thing, R->bad_thing.it[i])) {
                if (!first) first = R->bad_thing.it[i];
                ++rn;
            }
        int single_zhus = (rn == 1 && first && strstr(first, "\xe8\xaf\xb8\xe4\xba\x8b") != NULL);
        if (!single_zhus)
            for (int i = 0; i < R->bad_thing.n; ++i)
                if (cnl_l_has(&R->good_thing, R->bad_thing.it[i])) cnl_l_rm(&R->good_thing, R->bad_thing.it[i]);
    }
    /* 为空清理 */
    if (R->bad_thing.n == 0) cnl_l_push(&R->bad_thing, "\xe8\xaf\xb8\xe4\xba\x8b\xe4\xb8\x8d\xe5\xbf\x8c"); /* 诸事不忌 */
    if (R->good_thing.n == 0) cnl_l_push(&R->good_thing, "\xe8\xaf\xb8\xe4\xba\x8b\xe4\xb8\x8d\xe5\xae\x9c"); /* 诸事不宜 */
    /* 输出排序 */
    cnl_l_sort_stable(&R->bad_thing);
    cnl_l_sort_stable(&R->good_thing);
}


/* ------------------ 当日吉凶总评（与 almanac.js 评分规则一致） ------------------ */
/* score = 建除吉凶(±1) + 十二神吉凶(±1) + 宜忌等第≤1(±1)；
   score>=2 大吉，>=0 吉，>=-1 平，否则 凶 */
static void cnl_day_overall(cnlunar_result *R) {
    /* 建除吉：除 定 执 危 成 开；十二神吉：青龙 明堂 金贵 天德 玉堂 司命
       （almanac.js 作“金匮”，本库 cnl_12_DAYGOD 作“金贵”，与 today12day_name
       的黄道日判定 ecl∈{0,1,4,5,7,10} 一致）。针与草垛均为整字符 UTF-8
       （首字节只出现在字符边界），strstr 不会错位匹配。 */
    int score = 0;
    score += strstr("除定执危成开", R->today12day_officer) ? 1 : -1;
    score += strstr("青龙明堂金贵天德玉堂司命", R->today12day_god) ? 1 : -1;
    score += (R->thing_level <= 1) ? 1 : -1;
    const char *o = (score >= 2) ? "大吉" : (score >= 0) ? "吉" : (score >= -1) ? "平" : "凶";
    cnl_cpy(R->day_overall, o, 8);
}

/* ------------------ 星次 ------------------ */
static void cnl_east_zodiac(cnlunar_result *R) {
    int idx = ((R->next_solar_num + 23) % 24) / 2;
    cnl_cpy(R->today_east_zodiac, cnl_EAST_ZODIAC[idx], 8);
}

/* ------------------ 主流程 ------------------ */
static int cnl_run(cnlunar_result *R, cnl_scratch *cs) {
    if (R->year < 1901 || R->year > 2100) return CNLUNAR_ERR_RANGE;
    if (R->month < 1 || R->month > 12) return CNLUNAR_ERR_DATE;
    if (R->day < 1 || R->day > 31) return CNLUNAR_ERR_DATE;
    if (R->hour < 0 || R->hour > 23 || R->minute < 0 || R->minute > 59) return CNLUNAR_ERR_DATE;
    R->twohour_num = (R->hour + 1) / 2;

    int rc = cnl_lunar_date_num(R);
    if (rc) return rc;
    cnl_lunar_year_cn(R);
    cnl_lunar_month_cn(R);
    cnl_lunar_day_cn(R);
    cnl_phase_of_moon(R);
    cnl_today_solar_terms(R);
    int x = cnl_beginning_of_spring_x(R);
    cnl_8char(R, x);
    cnl_twohour(R);
    cnl_nums_and_season(R);
    cnl_12day_officer(R);
    cnl_zodiac(R);
    cnl_weekday_starzodiac(R);
    cnl_28stars(R);
    cnl_east_zodiac(R);
    cnl_peng_taboo(R);
    cnl_5elements(R);
    cnl_9flystar(R);
    cnl_lucky_gods_direction(R);
    cnl_fetal_god(R);
    cnl_twohour_lucky(R);
    cnl_meridians(R);
    /* 各瞬态清单复用同一片工作区：进入前把切取位置拨回起点 */
    if (cs) cs->pos = cs->base;
    cnl_holidays(R, cs);
    if (cs) cs->pos = cs->base;
    cnl_angel_demon(R, cs);
    cnl_day_overall(R);
    return CNLUNAR_OK;
}

static int cnl_calculate_common(cnlunar_result *out, int year, int month, int day,
                                int hour, int minute, unsigned options) {
    out->year = year; out->month = month; out->day = day;
    out->hour = hour; out->minute = minute;
    out->options = options;
    /* 宿主机便捷路径：工作区放本函数栈上（64KB，桌面栈足够大），
       保证 cnl_run 的 cs 恒非空（见 CNL_WS_LIST 注释） */
    unsigned char ws[CNLUNAR_WORKSPACE_MIN];
    cnl_scratch cs;
    cs.base = ws;
    cs.pos = ws;
    cs.end = ws + sizeof(ws);
    return cnl_run(out, &cs);
}

int cnlunar_calculate_ws(cnlunar_result *out, int year, int month, int day,
                         int hour, int minute, unsigned options,
                         void *workspace, size_t workspace_size) {
    if (!out) return CNLUNAR_ERR_NULL;
    if (!workspace || workspace_size < CNLUNAR_WORKSPACE_MIN) return CNLUNAR_ERR_WORKSPACE;
    /* 结果直接写入 out：不再使用 145KB 栈上临时结构 */
    memset(out, 0, sizeof(*out));
    out->year = year; out->month = month; out->day = day;
    out->hour = hour; out->minute = minute;
    out->options = options;
    cnl_scratch cs;
    cs.base = workspace;
    cs.pos = workspace;
    cs.end = (char *)workspace + workspace_size;
    return cnl_run(out, &cs);
}

int cnlunar_calculate(cnlunar_result *out, int year, int month, int day,
                      int hour, int minute, unsigned options) {
    if (!out) return CNLUNAR_ERR_NULL;
    /* 宿主机便捷版本：145KB 临时结果放本函数栈上，64KB 工作区由
       cnl_calculate_common 在栈上自备（桌面栈足够大）；
       嵌入式设备请使用 cnlunar_calculate_ws()，工作区由调用方提供（PSRAM）。 */
    cnlunar_result tmp;
    memset(&tmp, 0, sizeof(tmp));
    int rc = cnl_calculate_common(&tmp, year, month, day, hour, minute, options);
    if (rc == CNLUNAR_OK) *out = tmp;
    return rc;
}

#endif /* CNLUNAR_IMPLEMENTATION */
