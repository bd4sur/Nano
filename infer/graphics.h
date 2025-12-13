#ifndef __NANO_GFX_H__
#define __NANO_GFX_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

#define FONT_HEIGHT (12)
#define FONT_WIDTH_FULL (12)
#define FONT_WIDTH_HALF (6)

void gfx_init();
void gfx_close();
void gfx_refresh();

void fb_clear();
void fb_soft_clear();
void fb_plot(uint8_t x, uint8_t y, uint8_t mode);
void fb_draw_line(uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, uint8_t mode);
void fb_draw_circle(uint8_t x, uint8_t y, uint8_t r);
void fb_draw_char(uint8_t x, uint8_t y, uint8_t *glyph, uint8_t font_width, uint8_t font_height, uint8_t mode);
void fb_draw_bitmap(uint8_t x, uint8_t y, uint8_t sizex, uint8_t sizey, uint8_t BMP[], uint8_t mode);

void add_glyph_index_to_cache(uint32_t utf32, uint32_t index);
int32_t find_glyph_index_from_cache(uint32_t utf32);

uint8_t *get_glyph(uint32_t utf32, uint8_t *font_width, uint8_t *font_height);

#ifdef __cplusplus
}
#endif

#endif
