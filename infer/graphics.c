#include "graphics.h"

#include "display_hal.h"

#include "glyph.h"


void gfx_init(Nano_GFX *gfx, uint32_t width, uint32_t height, uint32_t color_mode) {
    // gfx = (Nano_GFX*)calloc(1, sizeof(Nano_GFX));
    gfx->color_mode = color_mode;
    gfx->width = width;
    gfx->height = height;
    gfx->frame_buffer_rgb888 = (uint8_t *)calloc(width * height * 3, sizeof(uint8_t));

    display_hal_init();

    gfx_clear(gfx);

}


void gfx_close(Nano_GFX *gfx) {
    // display_hal_close();
}


void gfx_refresh(Nano_GFX *gfx) {
    display_hal_refresh(gfx->frame_buffer_rgb888, gfx->width, gfx->height, 0, 0, gfx->width, gfx->height);
}

// 清屏函数
void gfx_clear(Nano_GFX *gfx) {
    memset(gfx->frame_buffer_rgb888, 0, gfx->width * gfx->height * 3);
    gfx_refresh(gfx);
}

// 清屏函数
void gfx_soft_clear(Nano_GFX *gfx) {
    memset(gfx->frame_buffer_rgb888, 0, gfx->width * gfx->height * 3);
    // gfx_refresh(gfx);
}

// 用纯白色填充整个屏幕
void gfx_fill_white(Nano_GFX *gfx) {
    memset(gfx->frame_buffer_rgb888, 255, gfx->width * gfx->height * 3);
    // gfx_refresh(gfx);
}

// 画点
// mode: 0-置黑  1-置色  2-异或  3-加色
void gfx_draw_point(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    if (x >= gfx->width || y >= gfx->height) {
        return;
    }

    uint32_t idx = (y * gfx->width + x) * 3;

    if (mode == 0) {
        gfx->frame_buffer_rgb888[idx+0] = 0;
        gfx->frame_buffer_rgb888[idx+1] = 0;
        gfx->frame_buffer_rgb888[idx+2] = 0;
    }
    else if (mode == 1) {
        gfx->frame_buffer_rgb888[idx+0] = MIN(255, red);
        gfx->frame_buffer_rgb888[idx+1] = MIN(255, green);
        gfx->frame_buffer_rgb888[idx+2] = MIN(255, blue);
    }
    else if (mode == 2) {
        uint8_t r = gfx->frame_buffer_rgb888[idx+0];
        uint8_t g = gfx->frame_buffer_rgb888[idx+1];
        uint8_t b = gfx->frame_buffer_rgb888[idx+2];

        gfx->frame_buffer_rgb888[idx+0] = (r == 0) ? 255 : 0;
        gfx->frame_buffer_rgb888[idx+1] = (g == 0) ? 255 : 0;
        gfx->frame_buffer_rgb888[idx+2] = (b == 0) ? 255 : 0;
    }
    else {
        uint8_t r = gfx->frame_buffer_rgb888[idx+0];
        uint8_t g = gfx->frame_buffer_rgb888[idx+1];
        uint8_t b = gfx->frame_buffer_rgb888[idx+2];

        gfx->frame_buffer_rgb888[idx+0] = MIN(255, r + red);
        gfx->frame_buffer_rgb888[idx+1] = MIN(255, g + green);
        gfx->frame_buffer_rgb888[idx+2] = MIN(255, b + blue);
    }
}

// 画线
// x1,y1:起点坐标
// x2,y2:结束坐标
void gfx_draw_line(Nano_GFX *gfx, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {

    // 垂直线
    if (x1 == x2 && y1 != y2) {
        int32_t delta = y2 - y1;
        for (int32_t y = y1; ((delta >= 0) ? (y <= y2) : (y >= y2)); ((delta >= 0) ? (y++) : (y--))) {
            gfx_draw_point(gfx, x1, y, red, green, blue, mode);
        }
    }
    // 水平线（或一点）
    else if (y1 == y2) {
        int32_t delta = x2 - x1;
        for (int32_t x = x1; ((delta >= 0) ? (x <= x2) : (x >= x2)); ((delta >= 0) ? (x++) : (x--))) {
            gfx_draw_point(gfx, x, y1, red, green, blue, mode);
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
            gfx_draw_point(gfx, uRow, uCol, red, green, blue, mode);
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

void gfx_draw_rectangle(Nano_GFX *gfx, uint32_t x0, uint32_t y0, uint32_t width, uint32_t height, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    for (uint32_t y = y0; y < MIN(gfx->height, y0 + height); y++) {
        for (uint32_t x = x0; x < MIN(gfx->width, x0 + width); x++) {
            gfx_draw_point(gfx, x, y, red, green, blue, mode);
        }
    }
}

// x,y:圆心坐标
// r:圆的半径
void gfx_draw_circle(Nano_GFX *gfx, uint32_t cx, uint32_t cy, uint32_t r, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    if (r == 0) {
        gfx_draw_point(gfx, cx, cy, red, green, blue, mode);
        return;
    }

    int32_t x = 0;
    int32_t y = r;
    int32_t d = 3 - 2 * r;  // 更稳健的初始决策参数（Bresenham 形式）

    while (x <= y) {
        // 定义8个对称点
        int32_t points[8][2] = {
            { cx + x, cy + y },
            { cx + y, cy + x },
            { cx - x, cy + y },
            { cx - y, cy + x },
            { cx + x, cy - y },
            { cx + y, cy - x },
            { cx - x, cy - y },
            { cx - y, cy - x }
        };

        for (int i = 0; i < 8; i++) {
            int32_t px = points[i][0];
            int32_t py = points[i][1];
            // 严格裁剪到屏幕范围
            if (px >= 0 && px < gfx->width && py >= 0 && py < gfx->height) {
                gfx_draw_point(gfx, (uint32_t)px, (uint32_t)py, red, green, blue, mode);
            }
        }

        if (d < 0) {
            d = d + 4 * x + 6;
        } else {
            d = d + 4 * (x - y) + 10;
            y--;
        }
        x++;
    }
}

// 显示汉字
// x,y:起点坐标
// mode:0,反色显示;1,正常显示
void gfx_draw_char(
    Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t *glyph,
    uint8_t font_width, uint8_t font_height,
    uint8_t red, uint8_t green, uint8_t blue, uint8_t mode
) {
    int32_t row_bytes = (font_height + 8 - 1) / 8;
    int32_t col_bytes = font_width;
    for (int32_t j = 0; j < row_bytes; j++) {
        int32_t bits = (j == (row_bytes-1)) ? (8 - ((8 * row_bytes) % font_height)) : 8;
        for (int32_t i = 0; i < col_bytes; i++) {
            uint8_t g = glyph[j * col_bytes + i];
            for (int32_t b = 0; b < bits; b++) {
                if ((g >> b) & 0x1) {
                    gfx_draw_point(gfx, (x+i), (y+j*8+b), red, green, blue, mode);
                }
                else if (mode == 0) {
                    gfx_draw_point(gfx, (x+i), (y+j*8+b), red, green, blue, !mode);
                }
            }
        }
    }
}


// 绘制一行文本
// mode: 1 - 正显，不修改底色
void gfx_draw_textline(Nano_GFX *gfx, wchar_t *line, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    uint32_t x_pos = x;
    uint32_t y_pos = y;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        uint32_t current_char = line[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        uint8_t *glyph = get_glyph(gfx, current_char, &font_width, &font_height);
        if (!glyph) {
            // printf("出现了字库之外的字符！\n");
            glyph = get_glyph(gfx, 12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        if (x_pos + font_width >= gfx->width) {
            break;
        }
        gfx_draw_char(gfx, x_pos, y_pos, glyph, font_width, font_height, red, green, blue, mode);
        x_pos += font_width;
    }
}


// 绘制一行文本，mode为1则为正显，为0则为反白
void gfx_draw_textline_centered(Nano_GFX *gfx, wchar_t *line, uint32_t cx, uint32_t cy, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {

    // 第一遍扫描：计算文本渲染长度
    int32_t total_width = 0;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        uint32_t current_char = line[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        uint8_t *glyph = get_glyph(gfx, current_char, &font_width, &font_height);
        if (!glyph) {
            glyph = get_glyph(gfx, 12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        total_width += font_width;
    }

    // 第二遍扫描：渲染
    int32_t x_pos = cx - (total_width/2);
    int32_t y_pos = cy - 6;
    if (y_pos < 0 || y_pos + 6 > gfx->height) return;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        uint32_t current_char = line[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        uint8_t *glyph = get_glyph(gfx, current_char, &font_width, &font_height);
        if (!glyph) {
            glyph = get_glyph(gfx, 12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        if (x_pos < 0) {
            x_pos += font_width;
            continue;
        }
        else if (x_pos + font_width > gfx->width) {
            break;
        }
        gfx_draw_char(gfx, x_pos, y_pos, glyph, font_width, font_height, red, green, blue, mode);
        x_pos += font_width;
    }
}


void gfx_draw_textline_mini(Nano_GFX *gfx, wchar_t *line, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    const uint8_t mini_glyph[43][7] = {
        {3, 5, 0x1F, 0x11, 0x1F, 0x00, 0x00}, // 0
        {3, 5, 0x12, 0x1F, 0x10, 0x00, 0x00}, // 1
        {3, 5, 0x1D, 0x15, 0x17, 0x00, 0x00}, // 2
        {3, 5, 0x15, 0x15, 0x1F, 0x00, 0x00}, // 3
        {3, 5, 0x0F, 0x08, 0x1F, 0x00, 0x00}, // 4
        {3, 5, 0x17, 0x15, 0x1D, 0x00, 0x00}, // 5
        {3, 5, 0x1F, 0x15, 0x1D, 0x00, 0x00}, // 6
        {3, 5, 0x01, 0x01, 0x1F, 0x00, 0x00}, // 7
        {3, 5, 0x1F, 0x15, 0x1F, 0x00, 0x00}, // 8
        {3, 5, 0x17, 0x15, 0x1F, 0x00, 0x00}, // 9

        {3, 5, 0x1E, 0x05, 0x1E, 0x00, 0x00}, // A
        {3, 5, 0x1F, 0x15, 0x0A, 0x00, 0x00}, // B
        {3, 5, 0x0E, 0x11, 0x11, 0x00, 0x00}, // C
        {3, 5, 0x1F, 0x11, 0x0E, 0x00, 0x00}, // D
        {3, 5, 0x1F, 0x15, 0x11, 0x00, 0x00}, // E
        {3, 5, 0x1F, 0x05, 0x11, 0x00, 0x00}, // F
        {3, 5, 0x0E, 0x11, 0x1D, 0x00, 0x00}, // G
        {3, 5, 0x1F, 0x04, 0x1F, 0x00, 0x00}, // H
        {3, 5, 0x11, 0x1F, 0x11, 0x00, 0x00}, // I
        {3, 5, 0x08, 0x10, 0x0F, 0x00, 0x00}, // J
        {4, 5, 0x1F, 0x04, 0x0A, 0x11, 0x00}, // K
        {3, 5, 0x1F, 0x10, 0x10, 0x00, 0x00}, // L
        {5, 5, 0x1F, 0x02, 0x04, 0x02, 0x1F}, // M
        {5, 5, 0x1F, 0x02, 0x04, 0x08, 0x1F}, // N
        {3, 5, 0x0E, 0x11, 0x0E, 0x00, 0x00}, // O
        {3, 5, 0x1F, 0x05, 0x02, 0x00, 0x00}, // P
        {5, 5, 0x0E, 0x11, 0x15, 0x09, 0x16}, // Q
        {3, 5, 0x1F, 0x05, 0x1A, 0x00, 0x00}, // R
        {3, 5, 0x12, 0x15, 0x09, 0x00, 0x00}, // S
        {3, 5, 0x01, 0x1F, 0x01, 0x00, 0x00}, // T
        {3, 5, 0x1F, 0x10, 0x1F, 0x00, 0x00}, // U
        {3, 5, 0x0F, 0x10, 0x0F, 0x00, 0x00}, // V
        {5, 5, 0x0F, 0x10, 0x0C, 0x10, 0x0F}, // W
        {3, 5, 0x1B, 0x04, 0x1B, 0x00, 0x00}, // X
        {3, 5, 0x03, 0x0C, 0x03, 0x00, 0x00}, // Y
        {3, 5, 0x19, 0x15, 0x13, 0x00, 0x00}, // Z

        {3, 5, 0x00, 0x00, 0x00, 0x00, 0x00}, // 空格
        {3, 5, 0x10, 0x00, 0x00, 0x00, 0x00}, // .
        {3, 5, 0x04, 0x04, 0x04, 0x00, 0x00}, // -
        {3, 5, 0x00, 0x0A, 0x00, 0x00, 0x00}, // :
        {3, 5, 0x00, 0x0E, 0x11, 0x00, 0x00}, // (
        {3, 5, 0x11, 0x0E, 0x00, 0x00, 0x00}, // )
        {5, 5, 0x13, 0x0B, 0x04, 0x1A, 0x19}  // %
    };

    uint32_t x_pos = x;
    uint32_t y_pos = y;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        wchar_t current_char = line[i];
        uint8_t font_width = 3;
        uint8_t font_height = 5;
        uint8_t *glyph = NULL;
        if (current_char >= L'0' && current_char <= L'9') {
            glyph = (uint8_t *)mini_glyph[current_char - L'0'];
        }
        else if (current_char >= L'A' && current_char <= L'Z') {
            glyph = (uint8_t *)mini_glyph[current_char - L'A' + 10];
        }
        else if (current_char == L' ') {
            glyph = (uint8_t *)mini_glyph[36 + 0];
        }
        else if (current_char == L'.') {
            glyph = (uint8_t *)mini_glyph[36 + 1];
        }
        else if (current_char == L'-') {
            glyph = (uint8_t *)mini_glyph[36 + 2];
        }
        else if (current_char == L':') {
            glyph = (uint8_t *)mini_glyph[36 + 3];
        }
        else if (current_char == L'(') {
            glyph = (uint8_t *)mini_glyph[36 + 4];
        }
        else if (current_char == L')') {
            glyph = (uint8_t *)mini_glyph[36 + 5];
        }
        else if (current_char == L'%') {
            glyph = (uint8_t *)mini_glyph[36 + 6];
        }
        else if (current_char == L'\n') {
            x_pos = x;
            y_pos += font_height + 1;
            continue;
        }
        else {
            // 未知字符，跳过
            x_pos += font_width;
            continue;
        }
        if (x_pos + font_width >= gfx->width) {
            break;
        }
        font_width = glyph[0];
        font_height = glyph[1];
        for (uint32_t i = 0; i < font_width; i++) {
            uint8_t column_data = glyph[2 + i];
            // printf("Column = %d\n", column_data);
            for (uint32_t j = 0; j < font_height; j++) {
                uint8_t m = ((column_data >> j) & 0x01) ? mode : (!mode);
                gfx_draw_point(gfx, x_pos + i, y_pos + j, red, green, blue, m);
            }
        }
        x_pos += (font_width + 1); // 字符间隔1像素
    }
}



// 绘制实心六边形
// 使用扫描线算法填充凸六边形
void gfx_draw_hexagon(
    Nano_GFX *gfx,
    uint32_t x1, uint32_t y1,
    uint32_t x2, uint32_t y2,
    uint32_t x3, uint32_t y3,
    uint32_t x4, uint32_t y4,
    uint32_t x5, uint32_t y5,
    uint32_t x6, uint32_t y6,
    uint8_t red, uint8_t green, uint8_t blue, uint8_t mode
) {
    // 将顶点存储到数组中便于处理
    uint32_t x_coords[6] = {x1, x2, x3, x4, x5, x6};
    uint32_t y_coords[6] = {y1, y2, y3, y4, y5, y6};
    
    // 计算六边形的垂直范围（限制在屏幕范围内）
    uint32_t y_min = y1;
    uint32_t y_max = y1;
    for (int i = 1; i < 6; i++) {
        if (y_coords[i] < y_min) y_min = y_coords[i];
        if (y_coords[i] > y_max) y_max = y_coords[i];
    }
    
    // 限制在画布范围内
    if (y_min >= gfx->height) return;
    if (y_max >= gfx->height) y_max = gfx->height - 1;
    
    // 对每一行扫描线，计算与六边形边的交点
    for (uint32_t y = y_min; y <= y_max; y++) {
        int32_t x_intersections[6];
        int num_intersections = 0;
        
        // 检查每条边与当前水平线的交点
        for (int i = 0; i < 6; i++) {
            uint32_t x_start = x_coords[i];
            uint32_t y_start = y_coords[i];
            uint32_t x_end = x_coords[(i + 1) % 6];
            uint32_t y_end = y_coords[(i + 1) % 6];
            
            // 跳过水平边（已在顶点处处理）
            if (y_start == y_end) continue;
            
            // 检查当前扫描线是否与这条边相交
            // 使用严格的包含-排除规则避免顶点重复计数
            int intersect = 0;
            if (y_start < y_end) {
                // 向上边：y_start <= y < y_end
                intersect = (y >= y_start && y < y_end) ? 1 : 0;
            } else {
                // 向下边：y_end < y <= y_start
                intersect = (y > y_end && y <= y_start) ? 1 : 0;
            }
            
            if (intersect) {
                // 线性插值计算交点的x坐标
                int32_t dy = (int32_t)y_end - (int32_t)y_start;
                int32_t dx = (int32_t)x_end - (int32_t)x_start;
                int32_t dy_scan = (int32_t)y - (int32_t)y_start;
                
                // x = x_start + dx * (y - y_start) / dy
                int32_t x_intersect = (int32_t)x_start + (dx * dy_scan) / dy;
                
                if (num_intersections < 6) {
                    x_intersections[num_intersections++] = x_intersect;
                }
            }
        }
        
        // 对交点进行排序（冒泡排序）
        for (int i = 0; i < num_intersections - 1; i++) {
            for (int j = i + 1; j < num_intersections; j++) {
                if (x_intersections[i] > x_intersections[j]) {
                    int32_t temp = x_intersections[i];
                    x_intersections[i] = x_intersections[j];
                    x_intersections[j] = temp;
                }
            }
        }
        
        // 填充交点之间的区域（凸多边形，成对填充）
        for (int i = 0; i < num_intersections; i += 2) {
            if (i + 1 >= num_intersections) break;
            
            int32_t x_start = x_intersections[i];
            int32_t x_end = x_intersections[i + 1];
            
            // 限制在画布范围内
            if (x_end < 0) continue;
            if (x_start >= (int32_t)gfx->width) continue;
            if (x_start < 0) x_start = 0;
            if (x_end >= (int32_t)gfx->width) x_end = gfx->width - 1;
            
            // 绘制扫描线段
            for (int32_t x = x_start; x <= x_end; x++) {
                gfx_draw_point(gfx, (uint32_t)x, y, red, green, blue, mode);
            }
        }
    }
}

uint8_t *get_glyph(Nano_GFX *gfx, uint32_t utf32, uint8_t *font_width, uint8_t *font_height) {
    if (utf32 < 32) {
        return NULL;
    }
    else if (utf32 >= 32 && utf32 < 127) {
        *font_width = 6;
        *font_height = 12;
        return ASCII_6_12[utf32 - 32];
    }
    else {
        int32_t index = binary_search(UTF32_LUT_SORTED, UTF32_LUT_INDEXS, GLYPH_CHAR_NUM, utf32);
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
