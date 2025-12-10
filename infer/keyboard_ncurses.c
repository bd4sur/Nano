#include "keyboard_hal.h"

#include <ncurses.h>

#include "platform.h"

int keyboard_hal_init() {
    return 0;
}

char keyboard_hal_read_key() {
    char ch = (char)getch();
    switch(ch) {
        case '1': return KEYCODE_NUM_1;
        case '2': return KEYCODE_NUM_2;
        case '3': return KEYCODE_NUM_3;
        case 'q': return KEYCODE_NUM_4;
        case 'w': return KEYCODE_NUM_5;
        case 'e': return KEYCODE_NUM_6;
        case 'a': return KEYCODE_NUM_7;
        case 's': return KEYCODE_NUM_8;
        case 'd': return KEYCODE_NUM_9;
        case 'z': return KEYCODE_NUM_STAR;
        case 'x': return KEYCODE_NUM_0;
        case 'c': return KEYCODE_NUM_HASH;
        case '4': return KEYCODE_NUM_A;
        case 'r': return KEYCODE_NUM_B;
        case 'f': return KEYCODE_NUM_C;
        case 'v': return KEYCODE_NUM_D;

        default: return KEYCODE_NUM_IDLE;
    }
}
