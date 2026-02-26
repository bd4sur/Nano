#ifndef __NANO_DISPLAY_HAL_H__
#define __NANO_DISPLAY_HAL_H__

#ifdef __cplusplus
extern "C" {
#endif


#include "platform.h"
#include "utils.h"

void display_hal_refresh(uint8_t *frame_buffer_rgb888, uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height);
void display_hal_init(void);
void display_hal_close(void);


#ifdef __cplusplus
}
#endif

#endif
