#include "graphics.h"

#include "display_hal.h"

#include "glyph.h"

// 定义宏以生成 stb 库的实现
#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "vendor/stb_image_resize2.h"

#include <string.h>

// 图像缓存项结构
typedef struct {
    char path[256];           // 图像路径（key）
    uint8_t *data;            // resize 后的图像数据（RGB888）
    uint32_t width;           // 图像宽度
    uint32_t height;          // 图像高度
    uint8_t valid;            // 此项是否有效
} ImageCacheEntry;

// 图像缓存数组（简单的固定大小缓存，LRU 淘汰）
static ImageCacheEntry s_image_cache[GFX_IMAGE_CACHE_SIZE];
static uint8_t s_cache_initialized = 0;

// 初始化缓存
static void init_image_cache(void) {
    if (s_cache_initialized) return;
    for (int i = 0; i < GFX_IMAGE_CACHE_SIZE; i++) {
        s_image_cache[i].path[0] = '\0';
        s_image_cache[i].data = NULL;
        s_image_cache[i].width = 0;
        s_image_cache[i].height = 0;
        s_image_cache[i].valid = 0;
    }
    s_cache_initialized = 1;
}

// 在缓存中查找图像，返回缓存项指针，未找到返回 NULL
static ImageCacheEntry* find_in_cache(const char *img_path, uint32_t width, uint32_t height) {
    for (int i = 0; i < GFX_IMAGE_CACHE_SIZE; i++) {
        if (s_image_cache[i].valid && 
            strcmp(s_image_cache[i].path, img_path) == 0 &&
            s_image_cache[i].width == width &&
            s_image_cache[i].height == height) {
            return &s_image_cache[i];
        }
    }
    return NULL;
}

// 将图像存入缓存（使用简单的 LRU 策略：将命中项移到最前，新项插入最前，淘汰最后）
static void add_to_cache(const char *img_path, uint8_t *data, uint32_t width, uint32_t height) {
    // 首先检查是否已存在（可能是 force_fetch 情况）
    ImageCacheEntry *existing = find_in_cache(img_path, width, height);
    if (existing) {
        // 更新现有缓存项的数据
        free(existing->data);
        existing->data = data;
        // 移到最前面（LRU）
        ImageCacheEntry temp = *existing;
        int idx = (int)(existing - s_image_cache);
        for (int i = idx; i > 0; i--) {
            s_image_cache[i] = s_image_cache[i - 1];
        }
        s_image_cache[0] = temp;
        return;
    }

    // 淘汰最后一个缓存项（如果已满）
    if (s_image_cache[GFX_IMAGE_CACHE_SIZE - 1].valid) {
        free(s_image_cache[GFX_IMAGE_CACHE_SIZE - 1].data);
    }

    // 将现有项后移
    for (int i = GFX_IMAGE_CACHE_SIZE - 1; i > 0; i--) {
        s_image_cache[i] = s_image_cache[i - 1];
    }

    // 新项插入最前
    s_image_cache[0].valid = 1;
    strncpy(s_image_cache[0].path, img_path, sizeof(s_image_cache[0].path) - 1);
    s_image_cache[0].path[sizeof(s_image_cache[0].path) - 1] = '\0';
    s_image_cache[0].data = data;
    s_image_cache[0].width = width;
    s_image_cache[0].height = height;
}


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

// 设置像素
inline void gfx_set_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b) {
    uint8_t *frame_buffer = gfx->frame_buffer_rgb888;
    uint32_t fb_width = gfx->width;
    uint32_t i = (y * fb_width + x) * 3;
    frame_buffer[ i ] = MIN(255, r);
    frame_buffer[i+1] = MIN(255, g);
    frame_buffer[i+2] = MIN(255, b);
}

// 叠加像素
inline void gfx_add_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b) {
    uint8_t *frame_buffer = gfx->frame_buffer_rgb888;
    uint32_t fb_width = gfx->width;
    uint32_t i = (y * fb_width + x) * 3;
    frame_buffer[ i ] = MIN(255, frame_buffer[ i ] + r);
    frame_buffer[i+1] = MIN(255, frame_buffer[i+1] + g);
    frame_buffer[i+2] = MIN(255, frame_buffer[i+2] + b);
}

// 数乘像素
inline void gfx_scale_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y, float k) {
    uint8_t *frame_buffer = gfx->frame_buffer_rgb888;
    uint32_t fb_width = gfx->width;
    uint32_t i = (y * fb_width + x) * 3;
    frame_buffer[ i ] = MIN(255, (uint8_t)(k * (float)frame_buffer[ i ]));
    frame_buffer[i+1] = MIN(255, (uint8_t)(k * (float)frame_buffer[i+1]));
    frame_buffer[i+2] = MIN(255, (uint8_t)(k * (float)frame_buffer[i+2]));
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


static inline float _fpart(float x) { return x - floorf(x); }
static inline float _rfpart(float x) { return 1.0f - _fpart(x); }

static void gfx_draw_line_xiaolin_wu(Nano_GFX *gfx,
    float x0, float y0, float x1, float y1,
    uint8_t r, uint8_t g, uint8_t b, uint8_t mode
) {
    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    int steep = fabsf(y1 - y0) > fabsf(x1 - x0);

    if (steep) {
        float t = x0; x0 = y0; y0 = t;
        t = x1; x1 = y1; y1 = t;
    }

    if (x0 > x1) {
        float t = x0; x0 = x1; x1 = t;
        t = y0; y0 = y1; y1 = t;
    }

    float dx = x1 - x0;
    float dy = y1 - y0;
    float gradient = (dx == 0.0f) ? 0.0f : dy / dx;

    // steep 决定算法坐标系与屏幕坐标系的映射关系，从而确定各自的合法边界
    int32_t x_limit = steep ? fb_height : fb_width;   // 算法 x 方向对应屏幕的宽度/高度
    int32_t y_limit = steep ? fb_width  : fb_height;  // 算法 y 方向对应屏幕的高度/宽度

    // handle first endpoint
    float xend = roundf(x0);
    float yend = y0 + gradient * (xend - x0);
    float xgap = _rfpart(x0 + 0.5f);
    int xpxl1 = (int)xend;
    int ypxl1 = (int)floorf(yend);
    if (xpxl1 >= 0 && xpxl1 < x_limit) {
        if (steep) {
            if (ypxl1 >= 0 && ypxl1 < y_limit) {
                gfx_add_pixel(gfx, ypxl1,   xpxl1, r * _rfpart(yend) * xgap, g * _rfpart(yend) * xgap, b * _rfpart(yend) * xgap);
            }
            if (ypxl1 + 1 >= 0 && ypxl1 + 1 < y_limit) {
                gfx_add_pixel(gfx, ypxl1+1, xpxl1, r * _fpart(yend)  * xgap, g * _fpart(yend)  * xgap, b * _fpart(yend)  * xgap);
            }
        } else {
            if (ypxl1 >= 0 && ypxl1 < y_limit) {
                gfx_add_pixel(gfx, xpxl1, ypxl1,   r * _rfpart(yend) * xgap, g * _rfpart(yend) * xgap, b * _rfpart(yend) * xgap);
            }
            if (ypxl1 + 1 >= 0 && ypxl1 + 1 < y_limit) {
                gfx_add_pixel(gfx, xpxl1, ypxl1+1, r * _fpart(yend)  * xgap, g * _fpart(yend)  * xgap, b * _fpart(yend)  * xgap);
            }
        }
    }
    float intery = yend + gradient;

    // handle second endpoint
    xend = roundf(x1);
    yend = y1 + gradient * (xend - x1);
    xgap = _fpart(x1 + 0.5f);
    int xpxl2 = (int)xend;
    int ypxl2 = (int)floorf(yend);
    if (xpxl2 >= 0 && xpxl2 < x_limit) {
        if (steep) {
            if (ypxl2 >= 0 && ypxl2 < y_limit) {
                gfx_add_pixel(gfx, ypxl2,   xpxl2, r * _rfpart(yend) * xgap, g * _rfpart(yend) * xgap, b * _rfpart(yend) * xgap);
            }
            if (ypxl2 + 1 >= 0 && ypxl2 + 1 < y_limit) {
                gfx_add_pixel(gfx, ypxl2+1, xpxl2, r * _fpart(yend)  * xgap, g * _fpart(yend)  * xgap, b * _fpart(yend)  * xgap);
            }
        } else {
            if (ypxl2 >= 0 && ypxl2 < y_limit) {
                gfx_add_pixel(gfx, xpxl2, ypxl2,   r * _rfpart(yend) * xgap, g * _rfpart(yend) * xgap, b * _rfpart(yend) * xgap);
            }
            if (ypxl2 + 1 >= 0 && ypxl2 + 1 < y_limit) {
                gfx_add_pixel(gfx, xpxl2, ypxl2+1, r * _fpart(yend)  * xgap, g * _fpart(yend)  * xgap, b * _fpart(yend)  * xgap);
            }
        }
    }

    // main loop：先对 x 范围做裁剪，消除循环变量本身的越界
    int x_start = xpxl1 + 1;
    int x_end = xpxl2 - 1;
    if (x_start < 0) x_start = 0;
    if (x_end >= x_limit) x_end = x_limit - 1;

    if (steep) {
        for (int x = x_start; x <= x_end; x++) {
            int y_base = (int)floorf(intery);
            if (y_base >= 0 && y_base < y_limit) {
                gfx_add_pixel(gfx, y_base,   x, r * _rfpart(intery), g * _rfpart(intery), b * _rfpart(intery));
            }
            if (y_base + 1 >= 0 && y_base + 1 < y_limit) {
                gfx_add_pixel(gfx, y_base+1, x, r * _fpart(intery),  g * _fpart(intery),  b * _fpart(intery));
            }
            intery += gradient;
        }
    } else {
        for (int x = x_start; x <= x_end; x++) {
            int y_base = (int)floorf(intery);
            if (y_base >= 0 && y_base < y_limit) {
                gfx_add_pixel(gfx, x, y_base,   r * _rfpart(intery), g * _rfpart(intery), b * _rfpart(intery));
            }
            if (y_base + 1 >= 0 && y_base + 1 < y_limit) {
                gfx_add_pixel(gfx, x, y_base+1, r * _fpart(intery),  g * _fpart(intery),  b * _fpart(intery));
            }
            intery += gradient;
        }
    }
}

void gfx_draw_line_anti_aliasing(Nano_GFX *gfx,
    float x1, float y1, float x2, float y2, float line_width,
    uint8_t r, uint8_t g, uint8_t b, uint8_t mode
) {
    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    if (line_width <= 0.0f) return;

    // 确保颜色在 [0, 255] 范围内
    uint8_t cr = MAX(0, MIN(255, r));
    uint8_t cg = MAX(0, MIN(255, g));
    uint8_t cb = MAX(0, MIN(255, b));

    // 线宽为1时，使用吴小林算法以获得更锐利的抗锯齿效果
    if (line_width <= 1.0f) {
        gfx_draw_line_xiaolin_wu(gfx, x1, y1, x2, y2, cr, cg, cb, mode);
        return;
    }

    float dx = x2 - x1;
    float dy = y2 - y1;
    float len_sq = dx * dx + dy * dy;

    // 退化为点：绘制圆形
    if (len_sq == 0.0f) { // TODO 与0比较
        float radius = line_width / 2.0f;
        float r_sq = radius * radius;
        int32_t xMin = MAX(0, (int32_t)floorf(x1 - radius));
        int32_t xMax = MIN(fb_width - 1, (int32_t)ceilf(x1 + radius));
        int32_t yMin = MAX(0, (int32_t)floorf(y1 - radius));
        int32_t yMax = MIN(fb_height - 1, (int32_t)ceilf(y1 + radius));

        for (int32_t y = yMin; y <= yMax; y++) {
            for (int32_t x = xMin; x <= xMax; x++) {
                float dist_sq = (float)((x - x1) * (x - x1) + (y - y1) * (y - y1));
                if (dist_sq <= r_sq) {
                    gfx_add_pixel(gfx, x, y, cr, cg, cb);
                }
            }
        }
        return;
    }

    float len = sqrtf(len_sq);
    float inv_len = 1.0f / len;
    float nx = -dy * inv_len; // 法向量（垂直于线段）
    float ny = dx * inv_len;

    (void)nx; (void)ny;

    float half_w = line_width / 2.0f;

    // 包围盒（含线宽）
    int32_t xMin = MAX(0, (int32_t)floorf(MIN(x1, x2) - half_w));
    int32_t xMax = MIN(fb_width - 1, (int32_t)ceilf(MAX(x1, x2) + half_w));
    int32_t yMin = MAX(0, (int32_t)floorf(MIN(y1, y2) - half_w));
    int32_t yMax = MIN(fb_height - 1, (int32_t)ceilf(MAX(y1, y2) + half_w));

    for (int32_t y = yMin; y <= yMax; y++) {
        for (int32_t x = xMin; x <= xMax; x++) {
            // 计算点 (x, y) 到线段的有符号距离
            int32_t px = x - x1;
            int32_t py = y - y1;

            // 投影长度（参数 t）
            float t = (float)(px * dx + py * dy) / len_sq;
            float closest_x = x1;
            float closest_y = y1;

            if (t < 0.0f) {
                closest_x = x1;
                closest_y = y1;
            }
            else if (t > 1.0f) {
                closest_x = x2;
                closest_y = y2;
            }
            else {
                closest_x = x1 + t * dx;
                closest_y = y1 + t * dy;
            }

            float dist = sqrtf((x - closest_x) * (x - closest_x) + (y - closest_y) * (y - closest_y));

            if (dist > half_w) continue;

            // 抗锯齿：边缘平滑过渡
            float alpha = 1.0f;
            float edge_fade = MIN(1.0f, half_w); // 自适应边缘宽度，防止细线中心像素过淡
            if (dist > half_w - edge_fade) {
                alpha = (half_w - dist) / edge_fade;
                alpha = MAX(0.0f, MIN(1.0f, alpha));
            }
            gfx_add_pixel(gfx, x, y, cr * alpha, cg * alpha, cb * alpha);
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
    Nano_GFX *gfx, uint32_t x, uint32_t y, const uint8_t *glyph,
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
                uint32_t px = x + i;
                uint32_t py = y + j*8 + b;
                if (px < 0 || px >= gfx->width || py < 0 || py >= gfx->height) continue;
                // if ((g >> b) & 0x1) {
                //     gfx_set_pixel(gfx, px, py, red, green, blue);
                // }
                // else if (mode == 0) {
                //     gfx_add_pixel(gfx, px, py, red, green, blue);
                // }
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
        const uint8_t *glyph = gfx_get_glyph(gfx, current_char, &font_width, &font_height);
        if (!glyph) {
            // printf("出现了字库之外的字符！\n");
            glyph = gfx_get_glyph(gfx, 12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
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
        const uint8_t *glyph = gfx_get_glyph(gfx, current_char, &font_width, &font_height);
        if (!glyph) {
            glyph = gfx_get_glyph(gfx, 12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
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
        const uint8_t *glyph = gfx_get_glyph(gfx, current_char, &font_width, &font_height);
        if (!glyph) {
            glyph = gfx_get_glyph(gfx, 12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
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
                if ((column_data >> j) & 0x01) {
                    gfx_draw_point(gfx, x_pos + i, y_pos + j, red, green, blue, mode);
                }
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

const uint8_t *gfx_get_glyph(Nano_GFX *gfx, uint32_t utf32, uint8_t *font_width, uint8_t *font_height) {
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

// 从指定文件读取图像并绘制到 frame_buffer（带缓存）
// img_path: 图像文件路径
// x0, y0: 目标区域左上角坐标
// width, height: 目标区域宽高（图像将被缩放到此大小）
// is_force_fetch: 如果非0，忽略缓存，重新读取并resize图像，并更新缓存
void gfx_draw_image(Nano_GFX *gfx, char *img_path, uint32_t x0, uint32_t y0, uint32_t width, uint32_t height, uint8_t is_force_fetch) {
    // 初始化缓存
    init_image_cache();
    
    uint8_t *draw_data = NULL;
    
    if (!is_force_fetch) {
        // 尝试从缓存获取
        ImageCacheEntry *cached = find_in_cache(img_path, width, height);
        if (cached != NULL) {
            draw_data = cached->data;
            // 更新 LRU：将此缓存项移到最前面
            int idx = (int)(cached - s_image_cache);
            if (idx > 0) {
                ImageCacheEntry temp = s_image_cache[idx];
                for (int i = idx; i > 0; i--) {
                    s_image_cache[i] = s_image_cache[i - 1];
                }
                s_image_cache[0] = temp;
            }
        }
    }
    
    // 缓存未命中或强制刷新：从文件读取并resize
    if (draw_data == NULL) {
        int img_w, img_h, channels;
        
        // 加载图像文件，获取 RGB 数据（3通道）
        unsigned char *img_data = stbi_load(img_path, &img_w, &img_h, &channels, 3);
        if (img_data == NULL) {
            // 加载失败，直接返回
            return;
        }
        
        // 分配缩放后图像的内存（RGB888 格式，3字节/像素）
        draw_data = (uint8_t *)malloc(width * height * 3);
        if (draw_data == NULL) {
            stbi_image_free(img_data);
            return;
        }
        
        // 缩放图像到目标尺寸
        stbir_resize_uint8_linear(
            img_data, img_w, img_h, 0,           // 输入图像数据、宽、高、行跨度（0表示连续）
            draw_data, (int)width, (int)height, 0,  // 输出图像数据、宽、高、行跨度
            STBIR_RGB                               // 像素布局：RGB
        );
        
        // 释放原始图像数据
        stbi_image_free(img_data);
        
        // 存入缓存（缓存接管内存所有权）
        add_to_cache(img_path, draw_data, width, height);
    }
    
    // 将图像绘制到 frame_buffer
    // 裁剪到 gfx 边界
    uint32_t x_end = (x0 + width > gfx->width) ? gfx->width : x0 + width;
    uint32_t y_end = (y0 + height > gfx->height) ? gfx->height : y0 + height;
    
    for (uint32_t y = y0; y < y_end; y++) {
        for (uint32_t x = x0; x < x_end; x++) {
            // 计算源图像中的像素位置
            uint32_t src_x = x - x0;
            uint32_t src_y = y - y0;
            uint32_t src_idx = (src_y * width + src_x) * 3;
            
            // 写入frame_buffer
            uint32_t idx = (y * gfx->width + x) * 3;
            gfx->frame_buffer_rgb888[idx+0] = MIN(255, draw_data[src_idx]);
            gfx->frame_buffer_rgb888[idx+1] = MIN(255, draw_data[src_idx + 1]);
            gfx->frame_buffer_rgb888[idx+2] = MIN(255, draw_data[src_idx + 2]);
        }
    }
}
