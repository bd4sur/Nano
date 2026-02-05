#ifndef __NANO_CELESTIAL_H__
#define __NANO_CELESTIAL_H__

#ifdef __cplusplus
extern "C" {
#endif

void dithering_fs(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height);
void dithering_fast(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height);

void draw_line(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float x1, float y1, float x2, float y2, float line_width, uint8_t r, uint8_t g, uint8_t b
);

// 绘制一行文本（居中）
void fb_draw_textline_centered(
    uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    wchar_t *line, int32_t cx, int32_t cy, uint8_t red, uint8_t green, uint8_t blue
);

void draw_circle(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float cx, float cy, float radius, uint8_t red, uint8_t green, uint8_t blue
);

void draw_rect(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float x0, float y0, float width, float height, uint8_t red, uint8_t green, uint8_t blue
);

void fb_draw_textline(
    uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    wchar_t *line, int32_t x, int32_t y, uint8_t red, uint8_t green, uint8_t blue
);

void render_sky(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float sky_radius, float center_x, float center_y,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double timezone, double longitude, double latitude,
    int32_t downsampling_factor,     // 降采样因子（设为0为自动，建议设为2）
    int32_t enable_opt_sym,          // 是否启用基于对称性的渲染优化（以画质为代价）
    int32_t enable_opt_lut,          // 是否启用基于对称性的渲染优化（以画质为代价）

    int32_t enable_equatorial_coord, // 是否启用赤道坐标圈
    int32_t enable_horizontal_coord, // 是否启用地平坐标圈
    int32_t enable_star_burst,       // 是否启用星芒效果
    int32_t enable_atmospheric_scattering, // 是否启用大气散射效果
    int32_t enable_star_name         // 是否显示天体名称
);

#ifdef __cplusplus
}
#endif

#endif
