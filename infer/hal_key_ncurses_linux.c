#include "hal_key.h"

#include <ncurses.h>

#include "nano_key_events.h"
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
    // 声明本 HAL 会向旁路队列投递按键边沿（供 (native Key) 本地库）
    nano_key_events_mark_source();
    return 0;
}

// 终端字节/特殊键 → NANO_KEY_* 逻辑键值（GUI 电平槽用；16 键重排见下方注释）
static uint8_t ncurses_map_key(int ch) {
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

    // 旁路投递（供 (native Key) 本地库）：终端没有 keyup/按住电平，一次 getch 就是
    // 一次离散按键，故按"按下沿 + 松开沿"成对投递（(Key.is_down) 在终端上恒 #f）。
    // 键码取【忠实值】——可打印字符就是 ASCII，其余用映射表的 NANO_KEY_*；刻意不做
    // GUI 的 16 键重排（'7'→NANO_KEY_1），否则 Scheme App 敲字母会得到错码。
    {
        uint8_t bypass = NANO_KEY_IDLE;
        if (ch >= 32 && ch <= 126) bypass = (uint8_t)ch;
        else bypass = ncurses_map_key(ch);
        if (bypass != NANO_KEY_IDLE) {
            nano_key_events_push(bypass, 1);
            nano_key_events_push(bypass, 0);
        }
    }

    return ncurses_map_key(ch);
}
