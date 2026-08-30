#ifndef __NANO_DISPLAY_HAL_H__
#define __NANO_DISPLAY_HAL_H__

#ifdef __cplusplus
extern "C" {
#endif


#include "platform.h"
#include "utils.h"

void display_hal_refresh(uint8_t *frame_buffer_rgb888, uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height);
void display_hal_refresh_rgb565(uint16_t *frame_buffer_rgb565, uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height);

void display_hal_refresh_rgb565_double(uint16_t *frame_buffer_rgb565_top, uint16_t *frame_buffer_rgb565_bottom,
    uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height);

void display_hal_init(void);
void display_hal_close(void);

void display_set_brightness(uint8_t value);

// LCD 控制器睡眠/唤醒（SLPIN/SLPOUT）。sleep 内含面板背光置0（Core2 物理切断 AXP192 DCDC3 /
// CoreS3 切断 BLDO1）；wakeup 恢复睡眠前由 setBrightness 记录的亮度。
// 注意：wakeup 后 ILI9342 需约 120ms 睡眠退出恢复时间，调用方应延时后再推帧。
void display_sleep(void);
void display_wakeup(void);


#ifdef __cplusplus
}
#endif

#endif
