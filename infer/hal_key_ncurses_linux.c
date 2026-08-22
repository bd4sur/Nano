#include "hal_key.h"

#include <ncurses.h>

#include "platform.h"
#include "hal_touch.h"

// 终端里鼠标与键盘共用同一条 stdin 字节流，只能由唯一的 getch 消费点解复用，
// 故此处捕获 KEY_MOUSE 并转发给触屏HAL（touch_ncurses.c）缓存——这是 ncurses
// 单输入流架构下不可避免的唯一耦合点。
extern void touch_ncurses_on_mouse(int32_t mx, int32_t my, uint32_t bstate);

// 触屏并行按键：鼠标按住期间，触点按屏幕上 4x4 宫格映射为虚拟按键，
// 与键盘输入互为备份（映射关系与 input_device_mp135.c 一致，
// 软键盘可见时框架会吞掉键盘区域的网格映射，改由软键盘接管，见 ui_app.c get_key_event）。
#define X0 (0)
#define X1 (SCREEN_WIDTH / 4 * 1)
#define X2 (SCREEN_WIDTH / 4 * 2)
#define X3 (SCREEN_WIDTH / 4 * 3)
#define X4 (SCREEN_WIDTH)
#define Y0 (0)
#define Y1 (SCREEN_HEIGHT / 4 * 1)
#define Y2 (SCREEN_HEIGHT / 4 * 2)
#define Y3 (SCREEN_HEIGHT / 4 * 3)
#define Y4 (SCREEN_HEIGHT)

static uint8_t map_touch_to_key(int x, int y) {
    if (y >= Y0 && y < Y1) {
        if (x >= X0 && x <  X1) return NANO_KEY_1;
        if (x >= X1 && x <  X2) return NANO_KEY_2;
        if (x >= X2 && x <  X3) return NANO_KEY_3;
        if (x >= X3 && x <= X4) return NANO_KEY_esc;
        else return NANO_KEY_IDLE;
    }
    else if (y >= Y1 && y < Y2) {
        if (x >= X0 && x <  X1) return NANO_KEY_4;
        if (x >= X1 && x <  X2) return NANO_KEY_5;
        if (x >= X2 && x <  X3) return NANO_KEY_6;
        if (x >= X3 && x <= X4) return NANO_KEY_shift;
        else return NANO_KEY_IDLE;
    }
    else if (y >= Y2 && y < Y3) {
        if (x >= X0 && x <  X1) return NANO_KEY_7;
        if (x >= X1 && x <  X2) return NANO_KEY_8;
        if (x >= X2 && x <  X3) return NANO_KEY_9;
        if (x >= X3 && x <= X4) return NANO_KEY_ctrl;
        else return NANO_KEY_IDLE;
    }
    else if (y >= Y3 && y <= Y4) {
        if (x >= X0 && x <  X1) return NANO_KEY_left;
        if (x >= X1 && x <  X2) return NANO_KEY_0;
        if (x >= X2 && x <  X3) return NANO_KEY_right;
        if (x >= X3 && x <= X4) return NANO_KEY_enter;
        else return NANO_KEY_IDLE;
    }
    else {
        return NANO_KEY_IDLE;
    }
}

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

    // 触屏作为并行按键输入：按住期间上报触点所在宫格的虚拟键码
    int32_t touch_x = 0, touch_y = 0, touch_pressed = 0;
    touch_read(&touch_x, &touch_y, &touch_pressed);
    if (touch_pressed) {
        return map_touch_to_key(touch_x, touch_y);
    }

    return NANO_KEY_IDLE;
}
