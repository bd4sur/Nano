/*
 * gen_almanac_tables.c —— 从 almanac.h（cnlunar C 实现）生成灵机 Scheme 数据表，
 * 并作为黄历结果的**参考输出器**（差分测试的黄金标准）。
 *
 * 用法：
 *   cc -O2 -I.. -o /tmp/gen_almanac tools/gen_almanac_tables.c
 *   /tmp/gen_almanac emit > app/lib/almanac_data.scm   # 生成 Scheme 数据表
 *   /tmp/gen_almanac ref <start_daynum> <count> <hour> <minute>   # 参考结果行
 *
 * 设计：本文件 #define CNLUNAR_IMPLEMENTATION 后直接 include almanac.h，因此在同一
 * 翻译单元内可以读到全部 static 表（ll000..ll191、神煞行、等第表…），逐项导出，
 * 无需解析 C 源码文本——表内容与 C 实现永远一致。
 *
 * 32 位宿主（wasm32）的整数载荷只有 27 位，所以：
 *   · cnl_SOLAR 的 48 位年份编码在此**解码**为 24 个 0..3 的偏移量；
 *   · cnl_LUNAR_MONTH 的位段解码为「12 个月大小 + 闰月号 + 闰月大小」；
 *   · cnl_TWOHOUR_LUCKY 的 16 位图解码为 12 个 0/1。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CNLUNAR_IMPLEMENTATION
#include "almanac.h"

/* ------------------------------------------------------------------ */
/* 工具                                                               */
/* ------------------------------------------------------------------ */

static void emit_str(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        if (*p == '"' || *p == '\\') { putchar('\\'); putchar(*p); }
        else putchar(*p);
    }
    putchar('"');
}

typedef struct { const char *const *a; int n; const char *name; } arr_def;

static void emit_arr(const arr_def *d) {
    printf("(define %s '(", d->name);
    for (int i = 0; i < d->n; ++i) {
        if (i) putchar(' ');
        emit_str(d->a[i]);
    }
    printf("))\n");
}

static void emit_int_arr(const char *name, const int *a, int n) {
    printf("(define %s '(", name);
    for (int i = 0; i < n; ++i) printf("%s%d", i ? " " : "", a[i]);
    printf("))\n");
}

#define NAMESZ(a) ((int)(sizeof(a) / sizeof((a)[0])))

/* ll 数组注册表：按名字排序（ll000..ll191） */
#define N_LL 193
static const char *const *ll_ptr[N_LL];
static int ll_n[N_LL];

static int ll_index_of(const char *const *p) {
    for (int i = 0; i < N_LL; ++i) if (ll_ptr[i] == p) return i;
    return -1;
}

/* 神煞行：输出 (name op base arg listIdx listN pairsIdx pairsN pstrIdx goodIdx goodN badIdx badN)
 * 索引字段用 -1 表示 NULL。 */
static void emit_row(const cnl_god_row *r) {
    int li = r->list ? ll_index_of(r->list) : -1;
    int gi = ll_index_of(r->good);
    int bi = ll_index_of(r->bad);
    int pi = -1;
    int si = r->pstr ? ll_index_of(r->pstr) : -1;
    if (r->pstr && si < 0) { fprintf(stderr, "row %s: pstr ptr not in registry\n", r->name); exit(1); }
    if (r->pairs) {
        static const int *tbl[6] = { pi000, pi001, pi002, pi003, pi004, pi005 };
        for (int i = 0; i < 6; ++i) if (r->pairs == tbl[i]) pi = i;
    }
    if (r->list && li < 0) { fprintf(stderr, "row %s: list ptr not in registry\n", r->name); exit(1); }
    if (gi < 0) { fprintf(stderr, "row %s: good ptr not in registry\n", r->name); exit(1); }
    if (bi < 0) { fprintf(stderr, "row %s: bad ptr not in registry\n", r->name); exit(1); }
    if (r->pairs && pi < 0) { fprintf(stderr, "row %s: pairs ptr unknown\n", r->name); exit(1); }
    printf("(");
    emit_str(r->name);
    printf(" %d ", r->op);
    if (r->base) emit_str(r->base); else printf("#f");
    printf(" %d %d %d %d %d %d %d %d %d %d)",
           r->arg, li, r->list_n, pi, r->pairs_n,
           si, gi, r->n_good, bi, r->n_bad);
}

/* ------------------------------------------------------------------ */
/* emit：生成 Scheme 数据表                                            */
/* ------------------------------------------------------------------ */

static void do_emit(void) {
    printf(";; ============================================================================\n");
    printf(";; almanac_data.scm —— 黄历数据表（由 tools/gen_almanac_tables.c 自动生成，勿手改）\n");
    printf(";;\n");
    printf(";; 真源：infer/almanac.h（cnlunar C 实现，MIT）。生成器直接 include 该头文件，\n");
    printf(";; 逐表导出，因此内容与 C 实现永远一致。\n");
    printf(";;\n");
    printf(";; 为适配 32 位灵机（整数载荷 27 位）：48 位的节气年表、位段编码的农历月表、\n");
    printf(";; 16 位的时辰吉凶位图都已在此**解码**为小整数列表。\n");
    printf(";; 数组一律用 quote 字面量（只读；运行期不得 push/set_item! 这些表）。\n");
    printf(";; ============================================================================\n\n");

    /* ---- 基础表 ---- */
    {
        arr_def a[] = {
            { cnl_SOLAR_TERMS_NAME, NAMESZ(cnl_SOLAR_TERMS_NAME), "SOLAR_TERMS_NAME" },
            { cnl_G60, NAMESZ(cnl_G60), "G60" },
            { cnl_STAR28, NAMESZ(cnl_STAR28), "STAR28" },
            { cnl_PENG_TABOO, NAMESZ(cnl_PENG_TABOO), "PENG_TABOO" },
            { cnl_12_DAYGOD, NAMESZ(cnl_12_DAYGOD), "DAYGOD" },
            { cnl_STAR_ZODIAC, NAMESZ(cnl_STAR_ZODIAC), "STAR_ZODIAC" },
            { cnl_DIRECTION, NAMESZ(cnl_DIRECTION), "DIRECTION" },
            { cnl_WEEKDAY, NAMESZ(cnl_WEEKDAY), "WEEKDAY" },
            { cnl_THINGS_SORT, NAMESZ(cnl_THINGS_SORT), "THINGS_SORT" },
            { cnl_FETAL_GOD, NAMESZ(cnl_FETAL_GOD), "FETAL_GOD" },
            { cnl_MERIDIANS, NAMESZ(cnl_MERIDIANS), "MERIDIANS" },
            { cnl_MERIDIAN_NOTE, NAMESZ(cnl_MERIDIAN_NOTE), "MERIDIAN_NOTE" },
            { cnl_MERIDIAN_YI, NAMESZ(cnl_MERIDIAN_YI), "MERIDIAN_YI" },
            { cnl_MERIDIAN_JI, NAMESZ(cnl_MERIDIAN_JI), "MERIDIAN_JI" },
            { cnl_BUJIANG, NAMESZ(cnl_BUJIANG), "BUJIANG" },
        };
        for (size_t i = 0; i < sizeof(a) / sizeof(a[0]); ++i) emit_arr(&a[i]);
        printf("\n");
    }

    /* 单字表：C 里是连写字符串，Scheme 侧拆成字符列表 */
    {
        const char *const *srcs[] = { NULL };
        (void)srcs;
    }
    {
        /* 地支/天干/八卦/十二建除：按 3 字节一个码点拆 */
        struct { const char *src; const char *name; } t[] = {
            { cnl_12_OFFICER_CHARS, "OFFICER_CHARS" },
            { cnl_TRIGRAM_8, "TRIGRAM8" },
            { cnl_STEM10_BASE, "STEM10" },
            { cnl_BRANCH12_BASE, "BRANCH12" },
        };
        for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); ++i) {
            const char *s = t[i].src;
            printf("(define %s '(", t[i].name);
            for (int k = 0; s[k * 3]; ++k) {
                if (k) putchar(' ');
                putchar('"');
                fwrite(s + k * 3, 1, 3, stdout);
                putchar('"');
            }
            printf("))\n");
        }
    }
    {
        /* 吉神方位表同样是逐支连写（10 个地支 × 1 字 = 10 个字符），
         * C 以 3*n 取第 n 个天干对应的字（表长 10，n=0..9）。 */
        static const char *const d0 = cnl_lucky_dir;
        static const char *const d1 = cnl_wealth_dir;
        static const char *const d2 = cnl_mascot_dir;
        static const char *const d3 = cnl_sunnoble_dir;
        static const char *const d4 = cnl_moonnoble_dir;
        const char *const srcs[5] = { d0, d1, d2, d3, d4 };
        const char *names[5] = { "LUCKY_DIR", "WEALTH_DIR", "MASCOT_DIR", "SUNNOBLE_DIR", "MOONNOBLE_DIR" };
        for (int i = 0; i < 5; ++i) {
            printf("(define %s '(", names[i]);
            for (int k = 0; srcs[i][k * 3]; ++k) {
                if (k) putchar(' ');
                putchar('"');
                fwrite(srcs[i] + k * 3, 1, 3, stdout);
                putchar('"');
            }
            printf("))\n");
        }
    }
    printf("\n");

    /* 星座分界（12 对 月/日，扁平） */
    {
        printf("(define STAR_ZODIAC_DATE '(");
        for (int i = 0; i < 12; ++i) printf("%s%d %d", i ? " " : "", cnl_STAR_ZODIAC_DATE[i][0], cnl_STAR_ZODIAC_DATE[i][1]);
        printf("))\n");
    }

    /* 节气编码向量（24） */
    emit_int_arr("ENC_VECTOR", cnl_ENC_VECTOR, NAMESZ(cnl_ENC_VECTOR));
    printf("\n");

    /* 节气年表：200 年 × 24 个偏移（解码自 48 位编码） */
    printf("(define SOLAR '(");
    for (int y = 0; y < 200; ++y) {
        unsigned long long data = cnl_SOLAR[y];
        for (int k = 0; k < 24; ++k) {
            int val = (int)((data >> (2 * k)) & 3ULL);
            printf("%s%d", (y || k) ? " " : "", val);
        }
    }
    printf("))\n");

    /* 农历年表：199 年 × (12 个月大小 30/29, 闰月号, 闰月大小 30/29) */
    printf("(define LUNAR '(");
    for (int y = 0; y < 199; ++y) {
        unsigned int tmp = cnl_LUNAR_MONTH[y];
        if (y) putchar(' ');
        printf("(");
        for (int m = 1; m <= 12; ++m)
            printf("%s%d", m > 1 ? " " : "", (tmp & (1u << (m - 1))) ? 30 : 29);
        int leap_month = (int)((tmp >> 13) & 0xf);
        int leap_len = leap_month ? (((tmp & (1u << 12)) ? 30 : 29)) : 0;
        printf(" %d %d)", leap_month, leap_len);
    }
    printf("))\n");

    /* 春节表：200 年 × (月, 日) */
    printf("(define NEWYEAR '(");
    for (int y = 0; y < 200; ++y) {
        unsigned int code = cnl_NEWYEAR[y];
        printf("%s%d %d", y ? " " : "", (int)((code >> 5) & 0x3), (int)(code & 0x1f));
    }
    printf("))\n");

    /* 时辰吉凶：60 日 × 12 位（1=凶，顺序 子..亥） */
    printf("(define TWOHOUR '(");
    for (int d = 0; d < 60; ++d) {
        unsigned short v = cnl_TWOHOUR_LUCKY[d];
        for (int j = 0; j < 12; ++j)
            printf("%s%d", (d || j) ? " " : "", (v & (1u << (11 - j))) ? 1 : 0);
    }
    printf("))\n\n");

    /* ---- ll 数组注册表 ---- */
    /* 192 个 ll 数组的登记（宏展开成 ptrs[数字] = ll00N; ...） */
    {
        const char *const *ptrs[N_LL];
        int ns[N_LL];
#define P(d, i) ptrs[i] = ll##d; ns[i] = NAMESZ(ll##d);
        P(000,0) P(001,1) P(002,2) P(003,3) P(004,4) P(005,5)
        P(006,6) P(007,7) P(008,8) P(009,9) P(010,10) P(011,11)
        P(012,12) P(013,13) P(014,14) P(015,15) P(016,16) P(017,17)
        P(018,18) P(019,19) P(020,20) P(021,21) P(022,22) P(023,23)
        P(024,24) P(025,25) P(026,26) P(027,27) P(028,28) P(029,29)
        P(030,30) P(031,31) P(032,32) P(033,33) P(034,34) P(035,35)
        P(036,36) P(037,37) P(038,38) P(039,39) P(040,40) P(041,41)
        P(042,42) P(043,43) P(044,44) P(045,45) P(046,46) P(047,47)
        P(048,48) P(049,49) P(050,50) P(051,51) P(052,52) P(053,53)
        P(054,54) P(055,55) P(056,56) P(057,57) P(058,58) P(059,59)
        P(060,60) P(061,61) P(062,62) P(063,63) P(064,64) P(065,65)
        P(066,66) P(067,67) P(068,68) P(069,69) P(070,70) P(071,71)
        P(072,72) P(073,73) P(074,74) P(075,75) P(076,76) P(077,77)
        P(078,78) P(079,79) P(080,80) P(081,81) P(082,82) P(083,83)
        P(084,84) P(085,85) P(086,86) P(087,87) P(088,88) P(089,89)
        P(090,90) P(091,91) P(092,92) P(093,93) P(094,94) P(095,95)
        P(096,96) P(097,97) P(098,98) P(099,99) P(100,100) P(101,101)
        P(102,102) P(103,103) P(104,104) P(105,105) P(106,106) P(107,107)
        P(108,108) P(109,109) P(110,110) P(111,111) P(112,112) P(113,113)
        P(114,114) P(115,115) P(116,116) P(117,117) P(118,118) P(119,119)
        P(120,120) P(121,121) P(122,122) P(123,123) P(124,124) P(125,125)
        P(126,126) P(127,127) P(128,128) P(129,129) P(130,130) P(131,131)
        P(132,132) P(133,133) P(134,134) P(135,135) P(136,136) P(137,137)
        P(138,138) P(139,139) P(140,140) P(141,141) P(142,142) P(143,143)
        P(144,144) P(145,145) P(146,146) P(147,147) P(148,148) P(149,149)
        P(150,150) P(151,151) P(152,152) P(153,153) P(154,154) P(155,155)
        P(156,156) P(157,157) P(158,158) P(159,159) P(160,160) P(161,161)
        P(162,162) P(163,163) P(164,164) P(165,165) P(166,166) P(167,167)
        P(168,168) P(169,169) P(170,170) P(171,171) P(172,172) P(173,173)
        P(174,174) P(175,175) P(176,176) P(177,177) P(178,178) P(179,179)
        P(180,180) P(181,181) P(182,182) P(183,183) P(184,184) P(185,185)
        P(186,186) P(187,187) P(188,188) P(189,189) P(190,190) P(191,191)
        ptrs[192] = cnl_noarr; ns[192] = 1;
#undef P
        for (int i = 0; i < N_LL; ++i) { ll_ptr[i] = ptrs[i]; ll_n[i] = ns[i]; }
    }
    printf("(define LL '(\n");
    for (int i = 0; i < N_LL; ++i) {
        printf("  (");  /* 每个数组一项 */
        for (int k = 0; k < ll_n[i]; ++k) {
            if (k) putchar(' ');
            emit_str(ll_ptr[i][k]);
        }
        printf(")\n");
    }
    printf("))\n\n");

    /* 神煞行引用的 ll 索引校验（count 必须 <= 数组长度） */
    for (int i = 0; i < 71; ++i) {
        const cnl_god_row *r = &cnl_ANGEL_ROWS[i];
        int li = r->list ? ll_index_of(r->list) : -1;
        if (li >= 0 && r->list_n > ll_n[li]) { fprintf(stderr, "angel %s list_n too big\n", r->name); exit(1); }
        if (r->n_good > ll_n[ll_index_of(r->good)]) { fprintf(stderr, "angel %s n_good too big\n", r->name); exit(1); }
        if (r->n_bad > ll_n[ll_index_of(r->bad)]) { fprintf(stderr, "angel %s n_bad too big\n", r->name); exit(1); }
    }
    for (int i = 0; i < 100; ++i) {
        const cnl_god_row *r = &cnl_DEMON_ROWS[i];
        int li = r->list ? ll_index_of(r->list) : -1;
        if (li >= 0 && r->list_n > ll_n[li]) { fprintf(stderr, "demon %s list_n too big\n", r->name); exit(1); }
        if (r->n_good > ll_n[ll_index_of(r->good)]) { fprintf(stderr, "demon %s n_good too big\n", r->name); exit(1); }
        if (r->n_bad > ll_n[ll_index_of(r->bad)]) { fprintf(stderr, "demon %s n_bad too big\n", r->name); exit(1); }
    }

    /* ---- 神煞行 ---- */
    printf("(define ANGEL '(\n");
    for (int i = 0; i < 71; ++i) { printf("  "); emit_row(&cnl_ANGEL_ROWS[i]); printf("\n"); }
    printf("))\n\n");
    printf("(define DEMON '(\n");
    for (int i = 0; i < 100; ++i) { printf("  "); emit_row(&cnl_DEMON_ROWS[i]); printf("\n"); }
    printf("))\n\n");

    /* ---- pi 对表 ---- */
    printf("(define PI '(");
    {
        static const int *tbl[6] = { pi000, pi001, pi002, pi003, pi004, pi005 };
        static const int tl[6] = { 3, 26, 6, 4, 4, 1 };
        for (int i = 0; i < 6; ++i) {
            if (i) putchar(' ');
            printf("(");
            for (int k = 0; k < tl[i]; ++k) printf("%s%d", k ? " " : "", tbl[i][k]);
            printf(")");
        }
    }
    printf("))\n\n");

    /* ---- 建除十二神宜忌 ---- */
    printf("(define OFFICER_THINGS '(\n");
    for (int i = 0; i < 12; ++i) {
        const cnl_officer_t *o = &cnl_officer_things[i];
        printf("  ("); emit_str(o->ch);
        printf(" %d %d %d %d)\n", ll_index_of(o->good), o->n_good, ll_index_of(o->bad), o->n_bad);
    }
    printf("))\n\n");

    /* ---- 日干宜忌 ---- */
    printf("(define D8THINGS '(\n");
    for (int i = 0; i < 16; ++i) {
        const cnl_d8thing_t *t = &cnl_day8_char_thing[i];
        printf("  ("); emit_str(t->ch);
        printf(" %d %d %d %d)\n", ll_index_of(t->good), t->n_good, ll_index_of(t->bad), t->n_bad);
    }
    printf("))\n\n");

    /* ---- 宜忌等第表 ---- */
    printf("(define THING_LEVEL '(\n");
    for (int i = 0; i < 11; ++i) {
        const cnl_tl_entry *e = &cnl_thing_level[i];
        printf("  ("); emit_str(e->key);
        printf(" (");
        for (int k = 0; k < e->n; ++k) {
            const cnl_tl_sub *s = &e->sub[k];
            if (k) putchar(' ');
            printf("("); emit_str(s->pat);
            printf(" %d %d %d)", ll_index_of(s->gods), s->n, s->level);
        }
        printf("))\n");
    }
    printf("))\n");
}

/* ------------------------------------------------------------------ */
/* ref：参考结果行（与 Scheme 侧 dump 相同格式）                        */
/* ------------------------------------------------------------------ */

static void print_list(const cnlunar_list *L) {
    for (int i = 0; i < L->n; ++i) {
        if (i) putchar('/');
        fputs(L->it[i], stdout);
    }
}

static int daynum_to_ymd(int days, int *y, int *m, int *d) {
    cnl_ymd(days, y, m, d);
    return 0;
}

static void do_ref(int start, int count, int hour, int minute) {
    static cnlunar_result R;
    static unsigned char ws[CNLUNAR_WORKSPACE_MIN];
    for (int i = 0; i < count; ++i) {
        int y, m, d;
        daynum_to_ymd(start + i, &y, &m, &d);
        memset(&R, 0, sizeof R);
        int rc = cnlunar_calculate_ws(&R, y, m, d, hour, minute, CNLUNAR_DEFAULT, ws, sizeof ws);
        printf("%d-%d-%d %d:%02d\t", y, m, d, hour, minute);
        if (rc != 0) { printf("ERR\n"); continue; }
        printf("%s%s%s%s\t", R.year8char, R.month8char, R.day8char, R.twohour8char);
        printf("%s\t%s\t%s\t%s\t%s\t%s\t", R.day_overall, R.week_day_cn, R.star_zodiac,
               R.today28star, R.today12day_officer, R.today12day_god);
        printf("%s\t%s\t%s\t%s\t", R.meridians, R.meridian_yi, R.meridian_ji, R.meridian_note);
        print_list(&R.good_thing); printf("\t");
        print_list(&R.bad_thing); printf("\t");
        printf("%s\t%s\t%s\t", R.the9flystar, R.fetal_god, R.peng_taboo);
        print_list(&R.twohour_lucky); printf("\t");
        print_list(&R.lucky_gods_direction); printf("\t");
        printf("%d,%d,%d,%d\t%d", R.lunar_year, R.lunar_month, R.lunar_day, R.is_leap_month,
               R.next_solar_num);
        printf("\t"); print_list(&R.good_god_name);
        printf("\t"); print_list(&R.bad_god_name);
        printf("\t%d\t%d", R.today_level, R.thing_level);
        printf("\n");
    }
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "emit") == 0) { do_emit(); return 0; }
    if (argc >= 6 && strcmp(argv[1], "ref") == 0) {
        do_ref(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
        return 0;
    }
    fprintf(stderr,
            "usage:\n  %s emit\n  %s ref <start_daynum> <count> <hour> <minute>\n",
            argv[0], argv[0]);
    return 2;
}
