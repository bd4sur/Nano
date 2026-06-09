#ifndef __NANO_GFX_H__
#define __NANO_GFX_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

// 图像缓存大小（最大缓存图像数量）
#ifndef GFX_IMAGE_CACHE_SIZE
#define GFX_IMAGE_CACHE_SIZE 4
#endif

#define FONT_HEIGHT (12)
#define FONT_WIDTH_FULL (12)
#define FONT_WIDTH_HALF (6)

#define GFX_COLOR_MODE_BINARY (1)
#define GFX_COLOR_MODE_RGB888 (11)
#define GFX_COLOR_MODE_RGB565 (12)

typedef struct Nano_GFX {
    uint32_t color_mode;

    // 通用帧缓冲
    uint8_t *frame_buffer_rgb888;
    uint16_t *frame_buffer_rgb565;

    uint16_t *frame_buffer_rgb565_top;
    uint16_t *frame_buffer_rgb565_bottom;

    uint32_t width;
    uint32_t height;

    uint8_t is_double_buffer;

    uint16_t *(*rgb565_access)(struct Nano_GFX *, uint32_t, uint32_t, uint32_t *);

    // 脏区域管理

    // 字库

    // 其他元数据

} Nano_GFX;

void gfx_test(Nano_GFX *gfx);

void gfx_init(Nano_GFX *gfx, uint32_t width, uint32_t height, uint32_t color_mode);
void gfx_close(Nano_GFX *gfx);
void gfx_refresh(Nano_GFX *gfx);

void gfx_set_brightness(Nano_GFX *gfx, int32_t brightness);

void gfx_clear(Nano_GFX *gfx);
void gfx_soft_clear(Nano_GFX *gfx);
void gfx_fill_white(Nano_GFX *gfx);

void gfx_get_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t *r, uint8_t *g, uint8_t *b);

void gfx_set_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b);
void gfx_add_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b);
void gfx_blend_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
void gfx_scale_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y, float k);
void gfx_reverse_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y);

void gfx_draw_point(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_line(Nano_GFX *gfx, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_line_anti_aliasing(Nano_GFX *gfx, float x1, float y1, float x2, float y2, float line_width, uint8_t r, uint8_t g, uint8_t b, uint8_t mode);
void gfx_draw_rectangle(Nano_GFX *gfx, uint32_t x0, uint32_t y0, uint32_t width, uint32_t height, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_circle(Nano_GFX *gfx, uint32_t cx, uint32_t cy, uint32_t r, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_circle_fill(Nano_GFX *gfx, uint32_t cx, uint32_t cy, uint32_t r, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_triangle(Nano_GFX *gfx, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, int32_t is_anti_aliasing, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_hexagon(Nano_GFX *gfx, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3, uint32_t x4, uint32_t y4, uint32_t x5, uint32_t y5, uint32_t x6, uint32_t y6, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_char(Nano_GFX *gfx, uint32_t x, uint32_t y, const uint8_t *glyph,
    uint8_t font_width, uint8_t font_height,
    uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);

void gfx_draw_textline(Nano_GFX *gfx, wchar_t *line, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_textline_centered(Nano_GFX *gfx, wchar_t *line, uint32_t cx, uint32_t cy, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_textline_mini(Nano_GFX *gfx, wchar_t *line, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);

void gfx_draw_busy(Nano_GFX *gfx);

const uint8_t *gfx_get_glyph(Nano_GFX *gfx, uint32_t utf32, uint8_t *font_width, uint8_t *font_height);

void gfx_gamma(Nano_GFX *gfx, float gamma);
void gfx_dithering(Nano_GFX *gfx);

void convert_rgb888_to_rgb565_double(Nano_GFX *gfx, uint8_t *rgb888, int32_t width, int32_t height);

void gfx_draw_image(Nano_GFX *gfx, char *img_path, uint32_t x0, uint32_t y0, uint32_t width, uint32_t height, uint8_t is_force_fetch);
void gfx_draw_image_buffer(Nano_GFX *gfx, uint8_t *img_buffer, uint32_t buffer_size, uint32_t x0, uint32_t y0, uint32_t width, uint32_t height);

int32_t gfx_decode_image_buffer(uint8_t *img_buffer, uint32_t buffer_size, uint32_t req_width, uint32_t req_height, uint8_t *out_rgb888, uint32_t *out_width, uint32_t *out_height);
void gfx_draw_rgb888_buffer(Nano_GFX *gfx, uint8_t *rgb888_buffer, uint32_t img_width, uint32_t img_height, uint32_t x0, uint32_t y0);

#ifdef __cplusplus
}
#endif

#endif
