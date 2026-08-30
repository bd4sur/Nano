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

// 推送一个水平带（A1 局部推帧）：buf 为半屏缓冲基址，half_base_y 为该半屏在整屏中的起始行。
// 满宽带（w == fb_width，文本滚动/菜单等典型场景）缓冲行连续，单次 pushPixels；
// 窄带（光标/小图标等）按行 setAddrWindow + pushPixels，不依赖窗口内自动换行语义。
static void display_push_band(uint16_t *buf, uint32_t fb_width, uint32_t half_base_y,
    uint32_t x0, uint32_t y0, uint32_t w, uint32_t rows) {
    if (w == fb_width) {
        display.setAddrWindow(x0, y0, w, rows);
        display.pushPixels(buf + ((y0 - half_base_y) * fb_width + x0), w * rows);
    } else {
        for (uint32_t r = 0; r < rows; r++) {
            display.setAddrWindow(x0, y0 + r, w, 1);
            display.pushPixels(buf + ((y0 + r - half_base_y) * fb_width + x0), w);
        }
    }
}

void display_hal_refresh_rgb565_double(uint16_t *frame_buffer_rgb565_top, uint16_t *frame_buffer_rgb565_bottom,
    uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height) {

    // A1 局部推帧：view 矩形与上/下半屏分别求交，分带推送（空带不产生 SPI 交易）
    if (view_width == 0 || view_height == 0) return;
    if (x0 >= fb_width || y0 >= fb_height) return;
    if (x0 + view_width  > fb_width)  view_width  = fb_width  - x0;
    if (y0 + view_height > fb_height) view_height = fb_height - y0;

    uint32_t half_height = fb_height / 2;
    uint32_t y1 = y0 + view_height;

    display.startWrite();          // 开始批量写入（提升性能）

    // 与上半屏 [0, half_height) 的交集
    if (y0 < half_height) {
        uint32_t band_end = (y1 < half_height) ? y1 : half_height;
        display_push_band(frame_buffer_rgb565_top, fb_width, 0, x0, y0, view_width, band_end - y0);
    }

    // 与下半屏 [half_height, fb_height) 的交集
    if (y1 > half_height) {
        uint32_t band_y0 = (y0 > half_height) ? y0 : half_height;
        display_push_band(frame_buffer_rgb565_bottom, fb_width, half_height, x0, band_y0, view_width, y1 - band_y0);
    }

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

void display_sleep(void) {
    display.sleep();   // 面板背光置0 + SLPIN（LGFX 内部 _brightness 记录不受影响）
}

void display_wakeup(void) {
    display.wakeup();  // SLPOUT + 恢复睡眠前亮度（不含 120ms 恢复延时，调用方负责）
}
