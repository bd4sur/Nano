#include "graphics.h"

#include "display_hal.h"

#include "glyph.h"

static uint8_t **FRAME_BUFFER = NULL;

static AVLNode *GLYPH = NULL;

static uint32_t *GLYPH_CACHE = NULL; // 字形缓存：前半部分是utf32，后半部分是字形index
static uint32_t GLYPH_CACHE_LEVEL = 0; // 字形缓存的水位

void gfx_init() {
    // 初始化GDRAM
    FRAME_BUFFER = (uint8_t **)calloc(FB_PAGES, sizeof(uint8_t *));
    for(uint8_t i = 0; i < FB_PAGES; i++) {
        FRAME_BUFFER[i] = (uint8_t *)calloc(FB_WIDTH, sizeof(uint8_t));
    }

    // 初始化字库
    GLYPH = buildAVLTree(UTF32_LUT, 7445);

    // 初始化字形缓存
    GLYPH_CACHE = (uint32_t*)calloc(2048, sizeof(uint32_t));
    GLYPH_CACHE_LEVEL = 0;

    display_hal_init();

    fb_clear();
}


void gfx_close() {
    display_hal_close();
    freeTree(GLYPH);
}


void gfx_refresh() {
    display_hal_refresh(FRAME_BUFFER);
}

// 清屏函数
void fb_clear(void) {
    for (uint8_t i = 0; i < FB_PAGES; i++) {
        for (uint8_t n = 0; n < FB_WIDTH; n++) {
            FRAME_BUFFER[i][n] = 0;
        }
    }
    gfx_refresh();
}

// 清屏函数
void fb_soft_clear(void) {
    for (uint8_t i = 0; i < FB_PAGES; i++) {
        for (uint8_t n = 0; n < FB_WIDTH; n++) {
            FRAME_BUFFER[i][n] = 0;
        }
    }
}

// 画点
// x: 0~127
// y: 0~63
// t: 0-置0  1-置1  2-异或
void fb_plot(uint8_t x, uint8_t y, uint8_t mode) {
    if (x >= FB_WIDTH || y >= FB_HEIGHT) {
        // printf("WARNING: Plot Coord Out of bound! Cancelled.\n");
        return;
    }
    uint8_t i, m, n;
    i = y / FB_PAGES;
    m = y % FB_PAGES;
    n = 1 << m;
    if (mode == 0) {
        FRAME_BUFFER[i][x] = ~FRAME_BUFFER[i][x];
        FRAME_BUFFER[i][x] |= n;
        FRAME_BUFFER[i][x] = ~FRAME_BUFFER[i][x];
    }
    else if (mode == 1) {
        FRAME_BUFFER[i][x] |= n;
    }
    else {
        FRAME_BUFFER[i][x] ^= n;
    }
}

// 画线
// x1,y1:起点坐标
// x2,y2:结束坐标
void fb_draw_line(uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, uint8_t mode) {

    // 垂直线
    if (x1 == x2 && y1 != y2) {
        int32_t delta = y2 - y1;
        for (int32_t y = y1; ((delta >= 0) ? (y <= y2) : (y >= y2)); ((delta >= 0) ? (y++) : (y--))) {
            fb_plot(x1, y, mode);
        }
    }
    // 水平线（或一点）
    else if (y1 == y2) {
        int32_t delta = x2 - x1;
        for (int32_t x = x1; ((delta >= 0) ? (x <= x2) : (x >= x2)); ((delta >= 0) ? (x++) : (x--))) {
            fb_plot(x, y1, mode);
        }
    }
    // 斜线
    else {
        int xerr = 0, yerr = 0, delta_x, delta_y, distance;
        int incx, incy, uRow, uCol;
        delta_x = x2 - x1; // 计算坐标增量
        delta_y = y2 - y1;
        uRow = x1; // 画线起点坐标
        uCol = y1;

        if (delta_x > 0)
            incx = 1; // 设置单步方向
        else if (delta_x == 0)
            incx = 0; // 垂直线
        else {
            incx = -1;
            delta_x = -delta_x;
        }

        if (delta_y > 0)
            incy = 1;
        else if (delta_y == 0)
            incy = 0; // 水平线
        else {
            incy = -1;
            delta_y = -delta_x;
        }

        if (delta_x > delta_y)
            distance = delta_x; // 选取基本增量坐标轴
        else
            distance = delta_y;

        for (int32_t t = 0; t < distance + 1; t++) {
            fb_plot(uRow, uCol, mode);
            xerr += delta_x;
            yerr += delta_y;
            if (xerr > distance) {
                xerr -= distance;
                uRow += incx;
            }
            if (yerr > distance) {
                yerr -= distance;
                uCol += incy;
            }
        }
    }
}

// x,y:圆心坐标
// r:圆的半径
void fb_draw_circle(uint8_t x, uint8_t y, uint8_t r)
{
    int a, b, num;
    a = 0;
    b = r;
    while (2 * b * b >= r * r) {
        fb_plot(x + a, y - b, 1);
        fb_plot(x - a, y - b, 1);
        fb_plot(x - a, y + b, 1);
        fb_plot(x + a, y + b, 1);

        fb_plot(x + b, y + a, 1);
        fb_plot(x + b, y - a, 1);
        fb_plot(x - b, y - a, 1);
        fb_plot(x - b, y + a, 1);

        a++;
        num = (a * a + b * b) - r * r; // 计算画的点离圆心的距离
        if (num > 0)
        {
            b--;
            a--;
        }
    }
}

// 显示汉字
// x,y:起点坐标
// num:汉字对应的序号
// mode:0,反色显示;1,正常显示
void fb_draw_char(uint8_t x, uint8_t y, uint8_t *glyph, uint8_t font_width, uint8_t font_height, uint8_t mode) {
    uint8_t x0 = x, y0 = y;
    uint32_t bytenum = (font_height / 8 + ((font_height % 8) ? 1 : 0)) * font_width; // 得到字体一个字符对应点阵集所占的字节数
    for (uint32_t i = 0; i < bytenum; i++) {
        uint8_t temp = glyph[i];
        uint8_t available_bits = (i >= bytenum - font_width) ? (8 - (bytenum / font_width) * 8 + font_height) : 8; // 只绘制有效位
        for (uint8_t m = 0; m < available_bits; m++) {
            if (temp & 0x01)
                fb_plot(x, y, mode);
            else
                fb_plot(x, y, !mode);
            temp >>= 1;
            y++;
        }
        x++;
        if ((x - x0) == font_width) {
            x = x0;
            y0 = y0 + 8;
        }
        y = y0;
    }
}

// x,y：起点坐标
// sizex,sizey,图片长宽
// BMP[]：要写入的图片数组
// mode:0,反色显示;1,正常显示
void fb_draw_bitmap(uint8_t x, uint8_t y, uint8_t sizex, uint8_t sizey, uint8_t BMP[], uint8_t mode) {
    uint32_t j = 0;
    uint8_t i, n, temp, m;
    uint8_t x0 = x, y0 = y;
    sizey = sizey / 8 + ((sizey % 8) ? 1 : 0);
    for (n = 0; n < sizey; n++)
    {
        for (i = 0; i < sizex; i++)
        {
            temp = BMP[j];
            j++;
            for (m = 0; m < 8; m++)
            {
                if (temp & 0x01)
                    fb_plot(x, y, mode);
                else
                    fb_plot(x, y, !mode);
                temp >>= 1;
                y++;
            }
            x++;
            if ((x - x0) == sizex)
            {
                x = x0;
                y0 = y0 + 8;
            }
            y = y0;
        }
    }
}

// 绘制一行文本，mode为1则为正显，为0则为反白
void fb_draw_textline(wchar_t *line, uint32_t x, uint32_t y, uint8_t mode) {
    uint32_t x_pos = x;
    uint32_t y_pos = y;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        uint32_t current_char = line[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        uint8_t *glyph = get_glyph(current_char, &font_width, &font_height);
        if (!glyph) {
            // printf("出现了字库之外的字符！\n");
            glyph = get_glyph(12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        if (x_pos + font_width >= 128) {
            break;
        }
        // NOTE 反色显示时，在每个字符场面额外补充一条线，避免菜单中高亮区域看起来顶格
        fb_draw_line(x_pos, y_pos - 1, x_pos+font_width-1, y_pos - 1, 1 - (mode % 2));
        fb_draw_char(x_pos, y_pos, glyph, font_width, font_height, (mode % 2));
        x_pos += font_width;
    }
}


void add_glyph_index_to_cache(uint32_t utf32, uint32_t index) {
    if (GLYPH_CACHE_LEVEL < 1024) {
        GLYPH_CACHE[   GLYPH_CACHE_LEVEL    ] = utf32;
        GLYPH_CACHE[GLYPH_CACHE_LEVEL + 1024] = index;
        GLYPH_CACHE_LEVEL++;
    }
}

int32_t find_glyph_index_from_cache(uint32_t utf32) {
    for (int32_t i = 0; i < GLYPH_CACHE_LEVEL; i++) {
        if (GLYPH_CACHE[i] == utf32) {
            return GLYPH_CACHE[i + 1024];
        }
    }
    return -1;
}

uint8_t *get_glyph(uint32_t utf32, uint8_t *font_width, uint8_t *font_height) {
    if(utf32 < 127) {
        *font_width = 6;
        *font_height = 12;
        return ASCII_6_12[utf32 - 32];
    }
    else {
/*
        uint32_t index = 0;
        // 首先从cache中取
        int32_t _index = find_glyph_index_from_cache(utf32);
        // 如果cache命中：直接取
        if (_index >= 0) {
            index = (uint32_t)_index;
        }
        // 如果cache不命中，则从AVL树中查询，并加入cache
        else {
            index = findIndex(GLYPH, utf32);
            add_glyph_index_to_cache(utf32, index);
        }
*/

        uint32_t index = findIndex(GLYPH, utf32);
        if (index >= 0 && index < 7445) {
            *font_width = 12;
            *font_height = 12;
            return GB2312_12_12[index];
        }
        else {
            return NULL;
        }
    }
}
