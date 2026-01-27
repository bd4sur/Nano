#ifndef __NANO_CELESTIAL_H__
#define __NANO_CELESTIAL_H__

#ifdef __cplusplus
extern "C" {
#endif

void init_glyph();

void dithering_fs(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height);
void dithering_fast(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height);

void render_sky(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float sky_radius, float center_x, float center_y,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double timezone, double longitude, double latitude
);

#ifdef __cplusplus
}
#endif

#endif
