#include "platform.h"
#include "keyboard_hal.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <linux/input.h>

#define INPUT_DEVICE "/dev/input/event0"

static int input_fd = -1;
static int touch_pressed = 0;
static int touch_x = 0;
static int touch_y = 0;

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
        if (x >= X0 && x <  X1) return KEYCODE_NUM_1;
        if (x >= X1 && x <  X2) return KEYCODE_NUM_2;
        if (x >= X2 && x <  X3) return KEYCODE_NUM_3;
        if (x >= X3 && x <= X4) return KEYCODE_NUM_A;
        else return KEYCODE_NUM_IDLE;
    }
    else if (y >= Y1 && y < Y2) {
        if (x >= X0 && x <  X1) return KEYCODE_NUM_4;
        if (x >= X1 && x <  X2) return KEYCODE_NUM_5;
        if (x >= X2 && x <  X3) return KEYCODE_NUM_6;
        if (x >= X3 && x <= X4) return KEYCODE_NUM_B;
        else return KEYCODE_NUM_IDLE;
    }
    else if (y >= Y2 && y < Y3) {
        if (x >= X0 && x <  X1) return KEYCODE_NUM_7;
        if (x >= X1 && x <  X2) return KEYCODE_NUM_8;
        if (x >= X2 && x <  X3) return KEYCODE_NUM_9;
        if (x >= X3 && x <= X4) return KEYCODE_NUM_C;
        else return KEYCODE_NUM_IDLE;
    }
    else if (y >= Y3 && y <= Y4) {
        if (x >= X0 && x <  X1) return KEYCODE_NUM_STAR;
        if (x >= X1 && x <  X2) return KEYCODE_NUM_0;
        if (x >= X2 && x <  X3) return KEYCODE_NUM_HASH;
        if (x >= X3 && x <= X4) return KEYCODE_NUM_D;
        else return KEYCODE_NUM_IDLE;
    }
    else {
        return KEYCODE_NUM_IDLE;
    }
}

int32_t keyboard_hal_init() {
    input_fd = open(INPUT_DEVICE, O_RDONLY | O_NONBLOCK);
    if (input_fd < 0) {
        return -1;
    }
    return 0;
}

uint8_t keyboard_hal_read_key() {
    struct input_event ev;

    // 读取所有当前可用的输入事件
    while (read(input_fd, &ev, sizeof(ev)) == sizeof(ev)) {
        if (ev.type == EV_ABS) {
            if (ev.code == ABS_X) {
                touch_x = ev.value;
            }
            else if (ev.code == ABS_Y) {
                touch_y = ev.value;
            }
        }
        else if (ev.type == EV_KEY && ev.code == BTN_TOUCH) {
            touch_pressed = ev.value;
        }
    }

    if (touch_pressed) {
        return map_touch_to_key(touch_x, touch_y);
    }

    return KEYCODE_NUM_IDLE;
}
