#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#define _XOPEN_SOURCE_EXTENDED
#include <ncurses.h>

#include "celestial.h"

// gcc utils.c celestial.c ephemeris.c sky.c -lncursesw -lm -o sky

#define FB_WIDTH (240)
#define FB_HEIGHT (240)
static uint8_t frame_buffer[FB_WIDTH * FB_HEIGHT * 4];

void draw_frame(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height) {
    printf("\033[?25l\033[H"); // 隐藏光标+定位到左上角
    for (int32_t y = 0; y < fb_height; y += 2) {
        for (int32_t x = 0; x < fb_width; x++) {
            int32_t i1 = (y * fb_width + x) * 4;
            int32_t i2 = ((y+1) * fb_width + x) * 4;
            uint8_t r1 = frame_buffer[i1];
            uint8_t g1 = frame_buffer[i1+1];
            uint8_t b1 = frame_buffer[i1+2];
            uint8_t r2 = ((y+1) < fb_height) ? frame_buffer[ i2 ] : 0;
            uint8_t g2 = ((y+1) < fb_height) ? frame_buffer[i2+1] : 0;
            uint8_t b2 = ((y+1) < fb_height) ? frame_buffer[i2+2] : 0;
            printf("\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm▀", r1,g1,b1, r2,g2,b2);
        }
        printf("\033[0m\n\r");
    }
    fflush(stdout);
}

int main(void) {
    initscr();
    cbreak();
    noecho();
    nodelay(stdscr, TRUE);
    keypad(stdscr, TRUE);

    printf("\033[?25l");
    fflush(stdout);

    uint64_t timestamp = 0;
    int32_t ch = 0;

    int32_t is_pause = 0;

    while ((ch = getch()) != 'q' && ch != 27) {
        if (ch == ' ') {
            is_pause = !is_pause;
        }

        if (!is_pause) {
            memset(frame_buffer, 0, FB_WIDTH * FB_HEIGHT * 4 * sizeof(uint8_t));

            double longitude = 119.0;
            double latitude = 31.0;
            double timezone = 8.0;

            int32_t second = timestamp % 60;
            int32_t minute = (timestamp / 60) % 60;
            int32_t hour = (timestamp / 3600) % 24;

            render_sky(frame_buffer, FB_WIDTH, FB_HEIGHT,
                120, 120, 120,
                2026, 1, 27, hour, minute, second,
                8.0, 119.0, 31.0);

            dithering_fs(frame_buffer, FB_WIDTH, FB_HEIGHT);

            timestamp += 60;

            draw_frame(frame_buffer, FB_WIDTH, FB_HEIGHT);
        }
    }

    // 恢复终端
    printf("\033[?25h\033[2J\033[H"); // 显示光标+清屏
    fflush(stdout);
    endwin();

    return 0;
}


