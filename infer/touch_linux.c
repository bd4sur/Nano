#include "platform.h"
#include "touch.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <linux/input.h>

// Linux 触屏HAL实现：基于 evdev（/dev/input/event0）轮询读取触屏状态。
// 与 input_device_mp135.c 一样，每次 touch_read  drain 所有可用的输入事件，
// 缓存最新的触点坐标与按下状态，边沿检测由上层自行实现。

#define INPUT_DEVICE "/dev/input/event0"

static int input_fd = -1;
static int touch_pressed = 0;
static int touch_x = 0;
static int touch_y = 0;

int32_t touch_init() {
    input_fd = open(INPUT_DEVICE, O_RDONLY | O_NONBLOCK);
    if (input_fd < 0) {
        return -1;
    }
    return 0;
}

int32_t touch_read(int32_t *x, int32_t *y, int32_t *is_pressed) {
    struct input_event ev;

    if (input_fd < 0) {
        return -1;
    }

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
        if (x) *x = touch_x;
        if (y) *y = touch_y;
        if (is_pressed) *is_pressed = 1;
    }
    else {
        if (is_pressed) *is_pressed = 0;
    }

    return 0;
}
