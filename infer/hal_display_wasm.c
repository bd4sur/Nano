// hal_display_wasm.c —— Nano-WASM-GUI 显示 HAL（Emscripten）
//
// 显示通路：gfx_refresh 推来的帧缓冲脏矩形（RGB888）在本 HAL 内展开为整屏
// RGBA8888 side buffer（C 侧展开，浏览器端零逐像素 JS 循环）；页面 JS 以
// requestAnimationFrame 节奏读取 side buffer（经 wasm_gui_get_fb_rgba 导出的
// 指针直接访问 wasm 线性内存），ImageData.data.set() + putImageData 上屏。
// JS 以 epoch 计数器（每次 refresh +1）判断有无新帧，静态画面零拷贝。

#include "hal_display.h"

#include <emscripten.h>

#define FB_WIDTH  (320)
#define FB_HEIGHT (240)

// 整屏 RGBA side buffer（静态分配，地址不变，供 JS 直接读）
static uint8_t  s_fb_rgba[FB_WIDTH * FB_HEIGHT * 4];
static uint32_t s_fb_epoch = 0;   // 刷新计数（每次 display_hal_refresh +1）

// ---- JS 侧访问接口（ccall/cwrap 用） ----

EMSCRIPTEN_KEEPALIVE
uint8_t *wasm_gui_get_fb_rgba(void) {
    return s_fb_rgba;
}

EMSCRIPTEN_KEEPALIVE
uint32_t wasm_gui_get_fb_epoch(void) {
    return s_fb_epoch;
}

EMSCRIPTEN_KEEPALIVE
uint32_t wasm_gui_get_fb_width(void) {
    return FB_WIDTH;
}

EMSCRIPTEN_KEEPALIVE
uint32_t wasm_gui_get_fb_height(void) {
    return FB_HEIGHT;
}

// ---- 显示 HAL ----

void display_hal_init(void) {
    // 画布初始化由页面 JS 完成（index.html），此处无事可做
}

void display_hal_refresh(
    uint8_t *frame_buffer_rgb888, uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height
) {
    if (fb_width != FB_WIDTH || fb_height != FB_HEIGHT) return;
    if (x0 >= FB_WIDTH || y0 >= FB_HEIGHT) return;
    if (x0 + view_width > FB_WIDTH)   view_width = FB_WIDTH - x0;
    if (y0 + view_height > FB_HEIGHT) view_height = FB_HEIGHT - y0;
    if (view_width == 0 || view_height == 0) return;

    // 脏区 RGB888 → RGBA 展开（C 速度，约 0.1ms/全屏）
    for (uint32_t row = 0; row < view_height; row++) {
        const uint8_t *src = frame_buffer_rgb888 + ((size_t)(y0 + row) * fb_width + x0) * 3;
        uint8_t *dst = s_fb_rgba + ((size_t)(y0 + row) * FB_WIDTH + x0) * 4;
        for (uint32_t col = 0; col < view_width; col++) {
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = 255;
            src += 3;
            dst += 4;
        }
    }
    s_fb_epoch++;
}

// RGB565 路径不参与（gfx 为 RGB888 模式，同其他 Linux 目标）
void display_hal_refresh_rgb565(
    uint16_t *frame_buffer_rgb565, uint32_t fb_width_in, uint32_t fb_height_in,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height
) {
    (void)frame_buffer_rgb565; (void)fb_width_in; (void)fb_height_in;
    (void)x0; (void)y0; (void)view_width; (void)view_height;
}

void display_hal_refresh_rgb565_double(uint16_t *frame_buffer_rgb565_top, uint16_t *frame_buffer_rgb565_bottom,
    uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height) {
    (void)frame_buffer_rgb565_top; (void)frame_buffer_rgb565_bottom;
    (void)fb_width; (void)fb_height;
    (void)x0; (void)y0; (void)view_width; (void)view_height;
}

void display_hal_close(void) {
    return;
}

void display_set_brightness(uint8_t value) {
    (void)value; // 网页端无物理背光
}

void display_sleep(void) {
    return;
}

void display_wakeup(void) {
    return;
}
