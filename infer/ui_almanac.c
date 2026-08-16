// ui_almanac.c - 黄历（农历择日）模态框
//
// 布局参照参考 HTML（almanac_container），并按 320x240 + 12px 字体的物理约束做最小适配：
//    容器 margin:10px auto / padding:3px / 底 #141516 / 圆角6，卡片间距3px、圆角4、底 #1a1a1a
//    卡1（2行）：八字(#ffaa66)+右侧当日吉凶徽标(#333底)  /  星期 | 星座 | 宿 | 建除 | 十二神(#66ccff)
//               （HTML 为单行 space-between，但 12px 字下 6 项+徽标需 ~385px > 可用 234px，拆两行）
//    卡2（2行）：当令器官(#fff200) / 宜(#99eb99) / 忌(#ffacc7) space-between + 说明居中(#808289)
//    卡3（2行）：宜 / 忌（标签彩色，值 #e2e5ef）
//    卡4（左列4行+右列3行）：九宫飞星 / 胎神 / 彭祖百忌 / 时辰吉凶 + 吉神方位(#aaff88，2+2+1 右对齐)
// 数据全部来自 almanac.h（cnlunar）。采用 workspace 版接口（PSRAM），任务栈零大分配。

#include "ui_almanac.h"
#include "utils.h"

#define CNLUNAR_IMPLEMENTATION
#include "almanac.h"

// ---- 模块持有的 PSRAM 内存（进入申请、关闭释放） ----
static cnlunar_result *s_result = NULL; // 145KB，platform_malloc(PSRAM)
static void           *s_ws     = NULL; // CNLUNAR_WORKSPACE_MIN，计算工作区
static int32_t         s_error  = 0;    // 1=错误态（日期越界/内存不足），draw 显示提示

// ---- 参考 HTML 配色 ----
#define AL_BG_R   (0x14) // #141516 容器底
#define AL_BG_G   (0x15)
#define AL_BG_B   (0x16)
#define AL_CARD_R (0x1a) // #1a1a1a 卡片底
#define AL_CARD_G (0x1a)
#define AL_CARD_B (0x1a)
#define AL_BAZI_R (255)  // #ffaa66 八字
#define AL_BAZI_G (170)
#define AL_BAZI_B (102)
#define AL_CYAN_R (102)  // #66ccff 星期/星座/宿/建除/十二神
#define AL_CYAN_G (204)
#define AL_CYAN_B (255)
#define AL_YEL_R  (255)  // #fff200 当令器官
#define AL_YEL_G  (242)
#define AL_YEL_B  (0)
#define AL_TXT_R  (226)  // #e2e5ef 常规灰白
#define AL_TXT_G  (229)
#define AL_TXT_B  (239)
#define AL_YI_R   (153)  // #99eb99 宜
#define AL_YI_G   (235)
#define AL_YI_B   (153)
#define AL_JI_R   (255)  // #ffacc7 忌
#define AL_JI_G   (172)
#define AL_JI_B   (199)
#define AL_DIR_R  (170)  // #aaff88 吉神方位
#define AL_DIR_G  (255)
#define AL_DIR_B  (136)
#define AL_BAD_R  (0x33) // #333333 当日吉凶徽标底
#define AL_BAD_G  (0x33)
#define AL_BAD_B  (0x33)
#define AL_GRAY_R (0x80) // #808289 注脚灰（说明/分隔符）
#define AL_GRAY_G (0x82)
#define AL_GRAY_B (0x89)

// 模态框内文本字体（信息密度大，用 12px 抗锯齿）
#define AL_FONT (GFX_FONT_ALPHA_12)

// 值文本缓冲上限（逐字符宽度不定，宽行会经 al_fit_w 裁到可用宽度）
#define AL_VAL_BUF (96)


// 将一行文本裁剪到可用宽度（超宽时从尾部截断，末尾补 "…"）
static void al_fit_w(Nano_GFX *gfx, wchar_t *s, int32_t max_w) {
    if (wcslen(s) == 0) return;
    if (gfx_font_measure_text(AL_FONT, s) <= max_w) return;
    /* 先整串截断，再逐字符回退给 "…" 腾位 */
    while (wcslen(s) > 1) {
        s[wcslen(s) - 1] = L'\0';
        wchar_t dot[2] = {L'…', 0};
        int32_t w_cur = gfx_font_measure_text(AL_FONT, s);
        int32_t w_dot = gfx_font_measure_text(AL_FONT, dot);
        if (w_cur + w_dot <= max_w) {
            swprintf(s + wcslen(s), 2, L"%ls", dot);
            break;
        }
    }
}

// 拼接清单为单行文本（空格分隔），并裁剪到可用宽度
static void al_list_line(Nano_GFX *gfx, wchar_t *dst, int32_t dst_cap,
                         const cnlunar_list *L, int32_t max_w) {
    wchar_t item[CNLUNAR_ITEM_MAX * 2];
    int32_t w = 0;
    int32_t first = 1;
    for (int32_t i = 0; i < L->n && w < dst_cap - 1; ++i) {
        _mbstowcs(item, L->it[i], CNLUNAR_ITEM_MAX);
        if (!first && w < dst_cap - 2) dst[w++] = L' ';
        int32_t il = (int32_t)wcslen(item);
        if (il > dst_cap - w - 1) il = dst_cap - w - 1;
        wmemcpy(dst + w, item, (size_t)il);
        w += il;
        first = 0;
    }
    dst[w] = L'\0';
    al_fit_w(gfx, dst, max_w);
}

// 绘制 “彩色标签 + 默认色值”（HTML 中标签 colored+bold、值继承 #e2e5ef；无粗体字库，bold 从略）
static void al_pair(Nano_GFX *gfx, int32_t x, int32_t y,
                    const wchar_t *label, const uint8_t lc[3],
                    const wchar_t *val) {
    gfx_font_draw_text(gfx, AL_FONT, (wchar_t *)label, x, y, lc[0], lc[1], lc[2], 1);
    int32_t lw = gfx_font_measure_text(AL_FONT, (wchar_t *)label);
    gfx_font_draw_text(gfx, AL_FONT, (wchar_t *)val, x + lw, y, AL_TXT_R, AL_TXT_G, AL_TXT_B, 1);
}

int32_t ui_almanac_open(int32_t year, int32_t month, int32_t day,
                        int32_t hour, int32_t minute) {
    if (s_result) return 0; // 已打开：幂等

    cnlunar_result *res = (cnlunar_result *)platform_malloc(sizeof(cnlunar_result));
    if (!res) { s_error = 1; return -1; }
    void *ws = platform_malloc(CNLUNAR_WORKSPACE_MIN);
    if (!ws) {
        free(res);
        s_error = 1;
        return -2;
    }
    int rc = cnlunar_calculate_ws(res, year, month, day, hour, minute,
                                  CNLUNAR_DEFAULT, ws, CNLUNAR_WORKSPACE_MIN);
    if (rc != CNLUNAR_OK) {
        free(ws);
        free(res);
        s_error = 1;   // 错误态：draw 显示提示（如 1900 年超出 cnlunar 数据范围）
        return rc;
    }
    s_result = res;
    s_ws = ws;
    s_error = 0;
    return 0;
}

void ui_almanac_close(void) {
    if (s_ws) { free(s_ws); s_ws = NULL; }
    if (s_result) { free(s_result); s_result = NULL; }
    s_error = 0;
}

int32_t ui_almanac_is_open(void) {
    return (s_result != NULL) || s_error;
}

void ui_almanac_draw(Nano_GFX *gfx) {
    if (s_error) {
        // 错误态：全屏提示
        gfx_draw_rectangle(gfx, 0, 0, gfx->width, gfx->height, AL_BG_R, AL_BG_G, AL_BG_B, 1);
        gfx_draw_rectangle(gfx, 6, 90, gfx->width - 12, 60, AL_CARD_R, AL_CARD_G, AL_CARD_B, 1);
        wchar_t msg[48];
        swprintf(msg, 48, L"该日期超出黄历计算范围");
        gfx_font_draw_text_centered(gfx, AL_FONT, msg, (int32_t)gfx->width / 2, 120, AL_TXT_R, AL_TXT_G, AL_TXT_B, 1);
        return;
    }
    if (!s_result) return;
    const cnlunar_result *R = s_result;
    const int32_t LH = gfx_font_line_height(AL_FONT);
    const int32_t W = (int32_t)gfx->width;    // 320
    const int32_t H = (int32_t)gfx->height;   // 240
    static const uint8_t CYAN[3]  = {AL_CYAN_R, AL_CYAN_G, AL_CYAN_B};
    static const uint8_t YI_C[3]  = {AL_YI_R, AL_YI_G, AL_YI_B};
    static const uint8_t JI_C[3]  = {AL_JI_R, AL_JI_G, AL_JI_B};
    static const uint8_t TXT_C[3] = {AL_TXT_R, AL_TXT_G, AL_TXT_B};

    // 容器底色（#141516，与容器同色，容器边界不外显）
    gfx_draw_rectangle(gfx, 0, 0, W, H, AL_BG_R, AL_BG_G, AL_BG_B, 1);

    /* 容器：margin:10px auto → x=10、宽300；padding:3px；卡片间距3px */
    const int32_t CX = 10 + 3;              // 卡片左沿
    const int32_t CW = W - 2 * 10 - 2 * 3;  // 卡片宽 294
    const int32_t GAP = 3;
    /* 卡高（HTML padding 换算）：卡1 pad6 两行+3px行距；卡2 pad4 两行+3px行距；卡3 pad4 两行；卡4 pad4 四行 */
    const int32_t card_h[4] = {6 + LH + 3 + LH + 6, 4 + LH + 3 + LH + 4, 4 + 2 * LH + 4, 4 + 4 * LH + 4};
    int32_t card_y[4];
    {
        int32_t y = 10 + 3;
        for (int32_t k = 0; k < 4; ++k) { card_y[k] = y; y += card_h[k] + GAP; }
    }
    for (int32_t k = 0; k < 4; ++k)
        gfx_draw_rectangle(gfx, CX, card_y[k], CW, card_h[k], AL_CARD_R, AL_CARD_G, AL_CARD_B, 1);

    const int32_t IX = CX + 6;      // 卡内左沿（卡1 padding 6）
    const int32_t IR = CX + CW - 6; // 卡内右沿

    // ---- 卡1：八字 + 当日吉凶徽标 / 星期 | 星座 | 宿 | 建除 | 十二神 ----
    {
        const int32_t ty1 = card_y[0] + 6;
        const int32_t ty2 = ty1 + LH + 3;

        // 当日吉凶徽标（行1 右侧，#333 底，padding 2px 8px）
        wchar_t badge[8];
        _mbstowcs(badge, R->day_overall, 7);
        int32_t bw = gfx_font_measure_text(AL_FONT, badge);
        int32_t brw = bw + 16;
        int32_t brx = IR - brw;
        gfx_draw_rectangle(gfx, brx, ty1 - 2, brw, LH + 4, AL_BAD_R, AL_BAD_G, AL_BAD_B, 1);
        static const uint8_t YEL_C[3] = {AL_YEL_R, AL_YEL_G, AL_YEL_B};
        const uint8_t *bc = (strcmp(R->day_overall, "\xe5\x87\xb6") == 0) ? JI_C   // 凶 → 粉
                          : (strcmp(R->day_overall, "\xe5\xb9\xb3") == 0) ? YEL_C  // 平 → 黄
                          : YI_C;                                                  // 大吉/吉 → 绿
        gfx_font_draw_text(gfx, AL_FONT, badge, brx + 8, ty1, bc[0], bc[1], bc[2], 1);

        // 行1：八字（橙，四柱连写）
        {
            wchar_t y8[12], m8[12], d8[12], h8[12];
            _mbstowcs(y8, R->year8char, 8);
            _mbstowcs(m8, R->month8char, 8);
            _mbstowcs(d8, R->day8char, 8);
            _mbstowcs(h8, R->twohour8char, 8);
            wchar_t bazi[48];
            swprintf(bazi, 48, L"%ls%ls%ls%ls", y8, m8, d8, h8);
            gfx_font_draw_text(gfx, AL_FONT, bazi, IX, ty1, AL_BAZI_R, AL_BAZI_G, AL_BAZI_B, 1);
        }

        // 行2：星期 | 星座 | 宿 | 建除 | 十二神（青），" | " 灰色分隔；超宽从尾部截断
        {
            wchar_t week[16], zod[16], xiu[12], jc[8], shen[8];
            _mbstowcs(week, R->week_day_cn, 12);
            _mbstowcs(zod, R->star_zodiac, 12);
            _mbstowcs(xiu, R->today28star, 11);
            _mbstowcs(jc, R->today12day_officer, 4);
            _mbstowcs(shen, R->today12day_god, 6);
            const wchar_t *segs[5] = {week, zod, xiu, jc, shen};
            const wchar_t *sep = L" | ";
            int32_t sep_w = gfx_font_measure_text(AL_FONT, (wchar_t *)sep);
            int32_t pen = IX;
            for (int32_t i = 0; i < 5; ++i) {
                if (i) {
                    if (pen + sep_w > IR) break;
                    gfx_font_draw_text(gfx, AL_FONT, (wchar_t *)sep, pen, ty2, AL_GRAY_R, AL_GRAY_G, AL_GRAY_B, 1);
                    pen += sep_w;
                }
                int32_t w = gfx_font_measure_text(AL_FONT, (wchar_t *)segs[i]);
                if (pen + w > IR) {
                    wchar_t tmp[24];
                    swprintf(tmp, 24, L"%ls", segs[i]);
                    al_fit_w(gfx, tmp, IR - pen);
                    gfx_font_draw_text(gfx, AL_FONT, tmp, pen, ty2, CYAN[0], CYAN[1], CYAN[2], 1);
                    break;
                }
                gfx_font_draw_text(gfx, AL_FONT, (wchar_t *)segs[i], pen, ty2, CYAN[0], CYAN[1], CYAN[2], 1);
                pen += w;
            }
        }
    }

    // ---- 卡2：子午流注（当令器官 / 宜 / 忌 space-between；说明居中灰字） ----
    {
        const int32_t inner = CW - 12;
        const int32_t r1y = card_y[1] + 4;
        const int32_t r2y = r1y + LH + 3;

        wchar_t org[12], myi[64], mji[64], note[AL_VAL_BUF];
        _mbstowcs(org, R->meridians, 8);
        _mbstowcs(myi, R->meridian_yi, 60);
        _mbstowcs(mji, R->meridian_ji, 60);
        _mbstowcs(note, R->meridian_note, 90);

        const wchar_t *lb_org = L"当令器官";
        int32_t w_lbo = gfx_font_measure_text(AL_FONT, (wchar_t *)lb_org);
        int32_t w_sp  = gfx_font_measure_text(AL_FONT, L" ");
        int32_t w_yi  = gfx_font_measure_text(AL_FONT, L"宜");
        int32_t w_ji  = gfx_font_measure_text(AL_FONT, L"忌");
        int32_t w_co  = gfx_font_measure_text(AL_FONT, L"：");
        int32_t g1 = w_lbo + w_sp + gfx_font_measure_text(AL_FONT, org);
        /* 宽度分配：当令器官短而固定；忌通常较短先满足（上限半幅），宜占剩余。
           值超宽时 al_fit_w 截断补 …（等价 HTML 的 overflow 隐藏） */
        int32_t budget = inner - g1 - 8;                      // 8 = 两个最小间隔
        al_fit_w(gfx, mji, budget / 2 - w_ji - w_co);
        int32_t g3 = w_ji + w_co + gfx_font_measure_text(AL_FONT, mji);
        al_fit_w(gfx, myi, budget - g3 - w_yi - w_co);
        int32_t g2 = w_yi + w_co + gfx_font_measure_text(AL_FONT, myi);
        int32_t gap = (inner - g1 - g2 - g3) / 2;
        if (gap < 4) gap = 4;
        int32_t x1 = IX, x2 = x1 + g1 + gap, x3 = x2 + g2 + gap;

        gfx_font_draw_text(gfx, AL_FONT, (wchar_t *)lb_org, x1, r1y, AL_TXT_R, AL_TXT_G, AL_TXT_B, 1);
        gfx_font_draw_text(gfx, AL_FONT, org, x1 + w_lbo + w_sp, r1y, AL_YEL_R, AL_YEL_G, AL_YEL_B, 1);
        // G2/G3：标签彩色 + 冒号默认色 + 值默认色（HTML 中仅标签 colored+bold）
        gfx_font_draw_text(gfx, AL_FONT, L"宜", x2, r1y, AL_YI_R, AL_YI_G, AL_YI_B, 1);
        gfx_font_draw_text(gfx, AL_FONT, L"：", x2 + w_yi, r1y, AL_TXT_R, AL_TXT_G, AL_TXT_B, 1);
        gfx_font_draw_text(gfx, AL_FONT, myi, x2 + w_yi + w_co, r1y, AL_TXT_R, AL_TXT_G, AL_TXT_B, 1);
        gfx_font_draw_text(gfx, AL_FONT, L"忌", x3, r1y, AL_JI_R, AL_JI_G, AL_JI_B, 1);
        gfx_font_draw_text(gfx, AL_FONT, L"：", x3 + w_ji, r1y, AL_TXT_R, AL_TXT_G, AL_TXT_B, 1);
        gfx_font_draw_text(gfx, AL_FONT, mji, x3 + w_ji + w_co, r1y, AL_TXT_R, AL_TXT_G, AL_TXT_B, 1);

        // 说明（居中，#808289）
        al_fit_w(gfx, note, inner);
        int32_t nw = gfx_font_measure_text(AL_FONT, note);
        gfx_font_draw_text(gfx, AL_FONT, note, IX + (inner - nw) / 2, r2y, AL_GRAY_R, AL_GRAY_G, AL_GRAY_B, 1);
    }

    // ---- 卡3：宜 / 忌 ----
    {
        const int32_t inner = CW - 12;
        const int32_t y1 = card_y[2] + 4;
        wchar_t val[AL_VAL_BUF];
        int32_t w_yi = gfx_font_measure_text(AL_FONT, L"宜：");
        int32_t w_ji = gfx_font_measure_text(AL_FONT, L"忌：");
        al_list_line(gfx, val, AL_VAL_BUF, &R->good_thing, inner - w_yi);
        al_pair(gfx, CX + 6, y1, L"宜：", YI_C, val);
        al_list_line(gfx, val, AL_VAL_BUF, &R->bad_thing, inner - w_ji);
        al_pair(gfx, CX + 6, y1 + LH, L"忌：", JI_C, val);
    }

    // ---- 卡4：左列 九宫飞星/胎神/彭祖百忌/时辰吉凶 + 右列吉神方位 ----
    {
        const int32_t inner = CW - 12;
        const int32_t ly = card_y[3] + 4;

        // 右列方位：2+2+1 三行（#aaff88，右对齐，垂直居中于 4 行空间）
        wchar_t dline[3][AL_VAL_BUF];
        int32_t dcnt[3] = {0, 0, 0};
        int32_t dw[3] = {0, 0, 0};
        {
            int32_t cnt = R->lucky_gods_direction.n;
            for (int32_t i = 0; i < cnt && i < 5; ++i) {
                int32_t row = (i < 2) ? 0 : (i < 4) ? 1 : 2;
                wchar_t it[CNLUNAR_ITEM_MAX + 1];
                _mbstowcs(it, R->lucky_gods_direction.it[i], CNLUNAR_ITEM_MAX);
                int32_t il = (int32_t)wcslen(it);
                int32_t *w = &dcnt[row];
                if (*w > 0 && *w < AL_VAL_BUF - 1) dline[row][(*w)++] = L' ';
                if (il > AL_VAL_BUF - *w - 1) il = AL_VAL_BUF - *w - 1;
                wmemcpy(dline[row] + *w, it, (size_t)il);
                *w += il;
            }
            for (int32_t r = 0; r < 3; ++r) {
                dline[r][dcnt[r]] = L'\0';
                dw[r] = gfx_font_measure_text(AL_FONT, dline[r]);
            }
        }
        int32_t dmax = dw[0];
        if (dw[1] > dmax) dmax = dw[1];
        if (dw[2] > dmax) dmax = dw[2];
        const int32_t left_w = inner - dmax - 8; // 左列可用宽（与右列留 8px）

        // 左列四行（标签+值均 #e2e5ef）
        wchar_t fly[16];
        {
            int32_t w = 0;
            for (int32_t i = 0; R->the9flystar[i] && w < 14; ++i) fly[w++] = (wchar_t)R->the9flystar[i];
            fly[w] = L'\0';
        }
        wchar_t fet[32], peng[64], hourluck[48];
        _mbstowcs(fet, R->fetal_god, 24);
        _mbstowcs(peng, R->peng_taboo, 60);
        {
            // 时辰吉凶：列今日吉时（13 项中第 13 项为明日子时，不显示）
            static const wchar_t *branches = L"子丑寅卯辰巳午未申酉戌亥";
            wchar_t hl[16];
            int32_t w = 0, any = 0;
            int32_t n = R->twohour_lucky.n; if (n > 12) n = 12;
            for (int32_t i = 0; i < n; ++i) {
                if (strcmp(R->twohour_lucky.it[i], "\xe5\x90\x89") == 0) { // 吉
                    if (w < 14) hl[w++] = branches[i];
                    any = 1;
                }
            }
            hl[w] = L'\0';
            if (any) swprintf(hourluck, 48, L"吉时 %ls", hl);
            else     swprintf(hourluck, 48, L"今日无吉时");
        }
        const wchar_t *labels[4] = {L"九宫飞星：", L"胎神：", L"彭祖百忌：", L"时辰吉凶："};
        wchar_t *values[4] = {fly, fet, peng, hourluck};
        for (int32_t i = 0; i < 4; ++i) {
            int32_t lw = gfx_font_measure_text(AL_FONT, (wchar_t *)labels[i]);
            al_fit_w(gfx, values[i], left_w - lw);
            al_pair(gfx, CX + 6, ly + i * LH, labels[i], TXT_C, values[i]);
        }

        // 右列方位（右对齐；3 行垂直居中于 4 行空间）
        for (int32_t r = 0; r < 3; ++r)
            if (dline[r][0])
                gfx_font_draw_text(gfx, AL_FONT, dline[r], IR - dw[r], ly + 7 + r * LH, AL_DIR_R, AL_DIR_G, AL_DIR_B, 1);
    }
}
