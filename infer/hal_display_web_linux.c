// hal_display_web_linux.c —— Nano-Web 显示 HAL
//
// C/S 分离架构的服务端显示端：gfx_refresh 推来的帧缓冲脏矩形（RGB888）经
// nano_ws_server 广播给全部网页客户端（薄客户端、远程桌面式推流，见 nano_ws_server.h）。
// 帧缓冲几何与 platform.h 的 NANO_WEB 平台块一致（320x240）。
//
// 线程安全：本函数由渲染线程（gfx_refresh）调用，nano_ws_server 内部自行同步。
// 背光/睡眠为无操作（网页端无物理背光；锁屏逻辑在 NANO_WEB 平台不参与编译）。

#include "hal_display.h"
#include "nano_ws_server.h"

#include <stdlib.h>

// 帧缓冲尺寸（与 platform.h NANO_WEB 块的 SCREEN_WIDTH/SCREEN_HEIGHT 一致；
// 不在此处引用 SCREEN_WIDTH 宏是为了让本文件可独立语法检查）
#define NANO_WEB_FB_WIDTH  (320)
#define NANO_WEB_FB_HEIGHT (240)

void display_hal_init(void) {
    uint16_t port = NANO_WEB_DEFAULT_PORT; // 默认 9002（platform.h NANO_WEB 块）
    const char *env = getenv("NANO_WEB_PORT");
    if (env != NULL && atoi(env) > 0) port = (uint16_t)atoi(env);
    if (nano_ws_server_start(port, NANO_WEB_FB_WIDTH, NANO_WEB_FB_HEIGHT) != 0) {
        printf("nano-web: FATAL: websocket server start failed (port %u in use?)\n", port);
    }
    else {
        printf("nano-web: web remote server listening on 0.0.0.0:%u (http://<host>:%u/)\n", port, port);
    }
}

void display_hal_refresh(
    uint8_t *frame_buffer_rgb888, uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height
) {
    nano_ws_server_push_frame(frame_buffer_rgb888, fb_width, fb_height,
                              x0, y0, view_width, view_height);
}

// NANO_WEB 的 gfx 色彩模式为 RGB888（main_init），RGB565 推帧路径不参与（同 ncurses HAL）
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
    nano_ws_server_stop();
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
