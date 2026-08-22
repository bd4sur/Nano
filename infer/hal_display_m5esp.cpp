#include "platform.h"
#include "hal_display.h"

#include <Arduino.h>
#include <esp32-hal-psram.h>
#include "M5Unified.h"

M5GFX display;

void display_hal_refresh(uint8_t *frame_buffer_rgb888, uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height
) {
    return;
}

void display_hal_refresh_rgb565(uint16_t *frame_buffer_rgb565, uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height
) {
    return;
}

void display_hal_refresh_rgb565_double(uint16_t *frame_buffer_rgb565_top, uint16_t *frame_buffer_rgb565_bottom,
    uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height) {

    uint32_t half_height = fb_height / 2;

    display.startWrite();          // 开始批量写入（提升性能）

    // 上半屏
    display.setAddrWindow(0, 0, fb_width, half_height);
    display.pushPixels(frame_buffer_rgb565_top, fb_width * half_height);

    // 下半屏
    display.setAddrWindow(0, half_height, fb_width, fb_height - half_height);
    display.pushPixels(frame_buffer_rgb565_bottom, fb_width * (fb_height - half_height));

    display.endWrite();            // 结束写入
}
void display_hal_init(void) {
    display = M5.Display;

    display.begin();
    // SPI写时钟按平台设置（DISPLAY_SPI_CLOCK_HZ 在 platform.h 中定义；0 = 使用 M5GFX 默认）
    // Core2 实测 60MHz 稳定（原装40MHz的1.5倍；80MHz出现闪屏/撕裂，信号完整性不足）；
    // CoreS3 未经真机验证，暂用默认值（DISPLAY_SPI_CLOCK_HZ = 0）。
#if DISPLAY_SPI_CLOCK_HZ > 0
    display.getPanel()->getBus()->setClock(DISPLAY_SPI_CLOCK_HZ);
#endif
    // display.setColorDepth(16);
    // display.setEpdMode(epd_mode_t::epd_fastest);
    display.setSwapBytes(true);
    display.setBrightness(204); // 全局默认背光（2026-08-01 由 255 调整为 204）

    display.clear();

    return;
}
void display_hal_close(void) {
    return;
}

void display_set_brightness(uint8_t value) {
    display.setBrightness(value);
}
