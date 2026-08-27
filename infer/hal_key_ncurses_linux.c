#include "hal_key.h"

#include <ncurses.h>

#include "platform.h"

// 终端里鼠标与键盘共用同一条 stdin 字节流，只能由唯一的 getch 消费点解复用，
// 故此处捕获 KEY_MOUSE 并转发给触屏HAL（touch_ncurses.c）缓存——这是 ncurses
// 单输入流架构下不可避免的唯一耦合点。
//
// 触屏 → 4x4 宫格虚拟按键的兼容映射曾置于此处，因旁路干净的触屏路径造成
// 架构混乱，已上移至输入事件层（ui_app.c ui_app_map_touch_to_grid16_key）；
// 本HAL只负责实体按键（终端键盘）。
extern void touch_ncurses_on_mouse(int32_t mx, int32_t my, uint32_t bstate);

int32_t input_device_init() {
    return 0;
}

uint8_t input_device_read_key() {
    // 一次性 drain 连续的鼠标事件：鼠标事件只更新触屏HAL缓存、不产生键码，
    // 逐帧渲染的场景每帧只调用一次本函数，逐个消费会让移动事件积压溢出输入队列
    int ch = getch();
    while (ch == KEY_MOUSE) {
        MEVENT ev;
        if (getmouse(&ev) == OK) {
            touch_ncurses_on_mouse(ev.x, ev.y, ev.bstate);
        }
        ch = getch();
    }
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
        case '\r': return NANO_KEY_enter;
        case KEY_BACKSPACE: return NANO_KEY_esc;
        case KEY_LEFT: return NANO_KEY_left;
        case KEY_RIGHT: return NANO_KEY_right;
        case KEY_UP: return NANO_KEY_up;
        case KEY_DOWN: return NANO_KEY_down;
        case KEY_ENTER: return NANO_KEY_enter;

        default: break;
    }

    return NANO_KEY_IDLE;
}
