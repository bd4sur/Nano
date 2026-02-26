#ifndef __NANO_GFX_H__
#define __NANO_GFX_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

#define FONT_HEIGHT (12)
#define FONT_WIDTH_FULL (12)
#define FONT_WIDTH_HALF (6)

#define GFX_COLOR_MODE_BINARY (1)
#define GFX_COLOR_MODE_RGB888 (11)
#define GFX_COLOR_MODE_RGB565 (12)

typedef struct {
    uint32_t color_mode;

    // 通用帧缓冲
    uint8_t *frame_buffer_rgb888;
    uint32_t width;
    uint32_t height;

    // 脏区域管理

    // 字库

    // 其他元数据

} Nano_GFX;

void gfx_init(Nano_GFX *gfx, uint32_t width, uint32_t height, uint32_t color_mode);
void gfx_close(Nano_GFX *gfx);
void gfx_refresh(Nano_GFX *gfx);

void gfx_clear(Nano_GFX *gfx);
void gfx_soft_clear(Nano_GFX *gfx);

void gfx_draw_point(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_line(Nano_GFX *gfx, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_circle(Nano_GFX *gfx, uint32_t cx, uint32_t cy, uint32_t r, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_char(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t *glyph,
    uint8_t font_width, uint8_t font_height,
    uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);

void gfx_draw_textline(Nano_GFX *gfx, wchar_t *line, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_textline_mini(Nano_GFX *gfx, wchar_t *line, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);

uint8_t *get_glyph(Nano_GFX *gfx, uint32_t utf32, uint8_t *font_width, uint8_t *font_height);

#ifdef __cplusplus
}
#endif

#endif
