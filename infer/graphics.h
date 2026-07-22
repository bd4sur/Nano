#ifndef __NANO_GFX_H__
#define __NANO_GFX_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

// 图像缓存大小（最大缓存图像数量）
#ifndef GFX_IMAGE_CACHE_SIZE
#define GFX_IMAGE_CACHE_SIZE 2
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
void gfx_draw_sector(Nano_GFX *gfx, int32_t x0, int32_t y0, int32_t r, int32_t angle0, int32_t angle1, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_hexagon(Nano_GFX *gfx, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3, uint32_t x4, uint32_t y4, uint32_t x5, uint32_t y5, uint32_t x6, uint32_t y6, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_char(Nano_GFX *gfx, uint32_t x, uint32_t y, const uint8_t *glyph,
    uint8_t font_width, uint8_t font_height,
    uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);

void gfx_draw_textline(Nano_GFX *gfx, wchar_t *line, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);

/* ============================================================================
 * 统一字体接口
 *
 * 支持多套字体（font_id）：
 *   GFX_FONT_BITMAP_12 - 12px 黑白二值点阵字模（glyph.h），行高13（字高12+间距1）
 *   GFX_FONT_ALPHA_12  - 12px alpha 平滑字模（glyph_12.h），行高15，基线12
 *   GFX_FONT_ALPHA_16  - 16px alpha 平滑字模（glyph_16.h），行高18，基线14
 *
 * 约定：
 *   - 同一套字体的行高固定（gfx_font_line_height），与具体字符无关；
 *   - 每个字符的宽度（笔位步进 x_advance）不定，必须逐字符用
 *     gfx_font_char_advance 获取（缺字时返回回退字符的宽度）；
 *   - 文本绘制位置均以“行顶”(y_top) 为基准，基线对齐由字体内部处理：
 *     baseline_y = y_top + gfx_font_baseline(font_id)。
 * ========================================================================== */

#define GFX_FONT_BITMAP_12 (0)
#define GFX_FONT_ALPHA_12  (1)
#define GFX_FONT_ALPHA_16  (2)

// alpha 字模包装函数（分别由 gfx_font_12.c / gfx_font_16.c 实现；
// 自动生成的 glyph_12.h / glyph_16.h 内部符号相互冲突，不能被同一编译单元包含）
uint8_t gfx_font_12_get_glyph(uint32_t codepoint, uint8_t *alpha,
                              uint8_t *w, uint8_t *h,
                              int8_t *x_offset, int8_t *y_offset, uint8_t *x_advance);
uint8_t gfx_font_16_get_glyph(uint32_t codepoint, uint8_t *alpha,
                              uint8_t *w, uint8_t *h,
                              int8_t *x_offset, int8_t *y_offset, uint8_t *x_advance);

// 字体整体参数：行高（固定）、基线（行顶到基线的距离）
int32_t gfx_font_line_height(uint32_t font_id);
int32_t gfx_font_baseline(uint32_t font_id);

// 逐字符笔位步进宽度（'\n' 为 0；缺字时返回回退字符的宽度）
int32_t gfx_font_char_advance(uint32_t font_id, uint32_t codepoint);

// 在 (x, y_top) 处绘制一个字符，返回实际笔位步进宽度（缺字时绘制回退字符）
// mode: 语义同 gfx_draw_point：0-置黑 1-置色 2-异或 3-加色 >=4-Alpha混合
int32_t gfx_font_draw_char(Nano_GFX *gfx, uint32_t font_id, uint32_t codepoint, int32_t x, int32_t y_top,
                           uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);

// 测量一行文本的渲染总宽度（逐字符实际宽度求和）
int32_t gfx_font_measure_text(uint32_t font_id, wchar_t *line);

// 绘制一行文本，(x, y_top) 为行框左上角；超宽截断
void gfx_font_draw_text(Nano_GFX *gfx, uint32_t font_id, wchar_t *line, int32_t x, int32_t y_top,
                        uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);

// 绘制一行水平垂直居中的文本，(cx, cy) 为行框中心
void gfx_font_draw_text_centered(Nano_GFX *gfx, uint32_t font_id, wchar_t *line, int32_t cx, int32_t cy,
                                 uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);

// ----------------------------------------------------------------------------
// 以下为 16px alpha 字模（GFX_FONT_ALPHA_16）的兼容接口，
// 实现已迁移到统一字体接口之上，新代码请直接使用 gfx_font_* 系列。
// ----------------------------------------------------------------------------
#define GFX_ALPHA_FONT_LINE_HEIGHT (18)
#define GFX_ALPHA_FONT_BASELINE    (14)
uint8_t gfx_get_glyph_alpha(uint32_t codepoint, uint8_t *alpha,
                            uint8_t *w, uint8_t *h,
                            int8_t *x_offset, int8_t *y_offset, uint8_t *x_advance);
int32_t gfx_draw_glyph_alpha(Nano_GFX *gfx, uint32_t codepoint, int32_t pen_x, int32_t baseline_y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
int32_t gfx_measure_textline_alpha(wchar_t *line);
void gfx_draw_textline_alpha(Nano_GFX *gfx, wchar_t *line, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
void gfx_draw_textline_alpha_centered(Nano_GFX *gfx, wchar_t *line, uint32_t cx, uint32_t cy, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode);
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
