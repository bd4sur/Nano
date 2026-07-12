#include "input_device.h"

#include <ncurses.h>

#include "platform.h"

int32_t input_device_init() {
    return 0;
}

uint8_t input_device_read_key() {
    int ch = getch();
    switch(ch) {
        case '0': return NANO_KEY_0;
        case '7': return NANO_KEY_1;
        case '8': return NANO_KEY_2;
        case '9': return NANO_KEY_3;
        case '4': return NANO_KEY_4;
        case '5': return NANO_KEY_5;
        case '6': return NANO_KEY_6;
        case '1': return NANO_KEY_7;
        case '2': return NANO_KEY_8;
        case '3': return NANO_KEY_9;
        case '*': return NANO_KEY_esc;
        case '-': return NANO_KEY_shift;
        case '+': return NANO_KEY_ctrl;
        case '\n': return NANO_KEY_enter;
        case KEY_BACKSPACE: return NANO_KEY_esc;
        case KEY_LEFT: return NANO_KEY_left;
        case KEY_RIGHT: return NANO_KEY_right;
        case KEY_UP: return NANO_KEY_up;
        case KEY_DOWN: return NANO_KEY_down;
        case KEY_ENTER: return NANO_KEY_enter;

        default: return NANO_KEY_IDLE;
    }
}
