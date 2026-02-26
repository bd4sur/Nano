#include "display_hal.h"

#define _XOPEN_SOURCE_EXTENDED
#include <ncurses.h>
#include <locale.h>
#include <string.h>
#include <unistd.h>

// 更新显存到frameBuffer
void display_hal_refresh(
    uint8_t *frame_buffer_rgb888, uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height
) {
    printf("\033[?25l\033[H"); // 隐藏光标+定位到左上角
    for (int32_t y = 0; y < fb_height; y += 2) {
        for (int32_t x = 0; x < fb_width; x++) {
            int32_t i1 = (y * fb_width + x) * 3;
            int32_t i2 = ((y+1) * fb_width + x) * 3;
            uint8_t r1 = frame_buffer_rgb888[i1];
            uint8_t g1 = frame_buffer_rgb888[i1+1];
            uint8_t b1 = frame_buffer_rgb888[i1+2];
            uint8_t r2 = ((y+1) < fb_height) ? frame_buffer_rgb888[ i2 ] : 0;
            uint8_t g2 = ((y+1) < fb_height) ? frame_buffer_rgb888[i2+1] : 0;
            uint8_t b2 = ((y+1) < fb_height) ? frame_buffer_rgb888[i2+2] : 0;
            printf("\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm▀", r1,g1,b1, r2,g2,b2);
        }
        printf("\033[0m\n\r");
    }
    fflush(stdout);


    // for (uint32_t p = 0; p < FB_PAGES; p++) {
    //     for (uint32_t x = 0; x < FB_WIDTH; x++) {
    //         for (uint8_t i = 0; i < 8; i++) {
    //             uint32_t y = p * 8 + i;
    //             if (FRAME_BUFFER[p][x] & (1 << i)) {
    //                 mvaddwstr(y, x, L"█");
    //             }
    //             else {
    //                 mvaddwstr(y, x, L" ");
    //             }
    //         }
    //     }
    // }
    // refresh();
}

// OLED的初始化
void display_hal_init(void) {
    initscr();
    curs_set(0);
    noecho();
    cbreak();
    timeout(0);

    if (LINES < FB_HEIGHT || COLS < FB_WIDTH) {
        endwin();
        return;
    }
}

void display_hal_close(void) {
    return;
}
