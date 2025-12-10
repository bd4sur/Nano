#include "display_hal.h"

#define _XOPEN_SOURCE_EXTENDED
#include <ncurses.h>
#include <locale.h>
#include <string.h>
#include <unistd.h>

// 更新显存到frameBuffer
void display_hal_refresh(uint8_t **FRAME_BUFFER) {
    for (uint32_t p = 0; p < FB_PAGES; p++) {
        for (uint32_t x = 0; x < FB_WIDTH; x++) {
            for (uint8_t i = 0; i < 8; i++) {
                uint32_t y = p * 8 + i;
                if (FRAME_BUFFER[p][x] & (1 << i)) {
                    mvaddwstr(y, x, L"█");
                }
                else {
                    mvaddwstr(y, x, L" ");
                }
            }
        }
    }
    refresh();
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
