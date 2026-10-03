#include "platform.h"
#include "hal_touch.h"

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

// TOUCH_SCALE：触屏坐标逆变换（编译期 -DTOUCH_SCALE=N 启用，默认 1，行为不变）。
// 与显示侧 FB_UPSCALE 配套（仅 Luckfox-Pico-86-Panel）：逻辑帧放大 N 倍并居中于
// 物理屏，故物理触点须先减黑边偏移再整除 N，还原为逻辑坐标；落在黑边区的触点
// 视为未按下。TOUCH_PHYS_WIDTH/HEIGHT 为 evdev ABS 物理量程（最大值+1），
// 默认取逻辑屏的 N 倍（即无黑边场景）。
#ifndef TOUCH_SCALE
#define TOUCH_SCALE 1
#endif
#ifndef TOUCH_PHYS_WIDTH
#define TOUCH_PHYS_WIDTH   (SCREEN_WIDTH  * TOUCH_SCALE)
#endif
#ifndef TOUCH_PHYS_HEIGHT
#define TOUCH_PHYS_HEIGHT  (SCREEN_HEIGHT * TOUCH_SCALE)
#endif
#if TOUCH_SCALE > 1
#define TOUCH_OFFSET_X  ((TOUCH_PHYS_WIDTH  - SCREEN_WIDTH  * TOUCH_SCALE) / 2)
#define TOUCH_OFFSET_Y  ((TOUCH_PHYS_HEIGHT - SCREEN_HEIGHT * TOUCH_SCALE) / 2)
#endif

static int input_fd = -1;
static int touch_pressed = 0;
static int touch_x = 0;
static int touch_y = 0;

int32_t touch_init() {
    // 设备节点可被 NANO_TOUCH_DEVICE 环境变量覆盖（PocketTerm35 的 Goodix 触屏
    // 节点号随枚举顺序变化，部署时用 /dev/input/by-path/ 稳定路径指定）
    const char *dev = getenv("NANO_TOUCH_DEVICE");
    if (dev == NULL || dev[0] == '\0') {
        dev = INPUT_DEVICE;
    }
    input_fd = open(dev, O_RDONLY | O_NONBLOCK);
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
        int out_x = touch_x;
        int out_y = touch_y;
#if TOUCH_SCALE > 1
        // 物理坐标 -> 逻辑坐标：先按物理量程裁剪黑边（避免负值整除误差），再整除缩放。
        // 注意不改动缓存值 touch_x/touch_y，避免多次读取被重复变换。
        if (out_x < TOUCH_OFFSET_X || out_x >= TOUCH_OFFSET_X + SCREEN_WIDTH  * TOUCH_SCALE ||
            out_y < TOUCH_OFFSET_Y || out_y >= TOUCH_OFFSET_Y + SCREEN_HEIGHT * TOUCH_SCALE) {
            if (is_pressed) *is_pressed = 0;
            return 0;
        }
        out_x = (out_x - TOUCH_OFFSET_X) / TOUCH_SCALE;
        out_y = (out_y - TOUCH_OFFSET_Y) / TOUCH_SCALE;
#endif
        if (x) *x = out_x;
        if (y) *y = out_y;
        if (is_pressed) *is_pressed = 1;
    }
    else {
        if (is_pressed) *is_pressed = 0;
    }

    return 0;
}
