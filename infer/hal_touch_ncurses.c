#include "platform.h"
#include "hal_touch.h"

#define _XOPEN_SOURCE_EXTENDED
#include <ncurses.h>

#include <stdio.h>

// tty 目标触屏HAL实现：以终端鼠标上报模拟触屏。
// 终端里鼠标与键盘共用 ncurses 的 getch 输入流，本文件无法独立读取，
// 故鼠标事件由 input_device_ncurses.c 在 getch 返回 KEY_MOUSE 时捕获，
// 经 touch_ncurses_on_mouse() 转发到这里的缓存，touch_read() 仅读取缓存
// （与 touch_linux.c 的 drain+缓存 模式等价，边沿检测仍由上层实现）。
//
// 坐标映射：display_ncurses.c 以半块字符 ▀ 渲染 320x240 帧缓冲，
// 1 个字符列 = 1 个像素，1 个字符行 = 上下 2 个像素行，
// 故像素坐标 = (列, 行 * 2)，并裁剪到屏幕范围内。
//
// 注意：开启鼠标上报后终端原生选中/复制被程序接管（多数终端可按住 Shift 绕过）。

#define FB_WIDTH  320
#define FB_HEIGHT 240

static int32_t s_pressed = 0;
static int32_t s_x = 0;
static int32_t s_y = 0;

// 供 input_device_ncurses.c 在收到 KEY_MOUSE 时调用，缓存最新鼠标状态。
// mx / my 为字符单元格坐标；bstate 为 ncurses 的鼠标按键状态掩码。
void touch_ncurses_on_mouse(int32_t mx, int32_t my, uint32_t bstate) {
    // 注意：ncurses 对“按住拖动”只上报 REPORT_MOUSE_POSITION（不带 BUTTON1_PRESSED 位），
    // 故拖动是否有效取决于此处已锁存的按下状态，而不是 bstate 中的按键位
    if (bstate & BUTTON1_RELEASED) {
        s_pressed = 0;
    }
    else if (bstate & BUTTON1_PRESSED) {
        s_pressed = 1;
    }
    else if (!(bstate & REPORT_MOUSE_POSITION) || !s_pressed) {
        return; // 滚轮事件、或未按下时的悬停移动：不更新坐标
    }

    if (mx < 0) mx = 0;
    if (mx > FB_WIDTH - 1) mx = FB_WIDTH - 1;
    if (my < 0) my = 0;
    if (my > FB_HEIGHT / 2 - 1) my = FB_HEIGHT / 2 - 1;
    s_x = mx;
    s_y = my * 2;
}

int32_t touch_init() {
    // 左键按下/松开 + 位置上报（REPORT_MOUSE_POSITION 使拖动时持续更新坐标）；
    // 显式开启 1002 button-motion（仅按住时上报移动）与 SGR 1006 扩展坐标格式：
    //   - 不用 1003 any-motion：悬停移动也会上报，FLIP/玲珑仪等逐帧渲染的场景
    //     每帧只消费一个事件，悬停事件会积压溢出输入队列、冲掉真正的按下事件；
    //   - 1006 避免列号超过 223 时经典协议坐标溢出
    mousemask(BUTTON1_PRESSED | BUTTON1_RELEASED | REPORT_MOUSE_POSITION, NULL);
    mouseinterval(0); // 不做双击判定延迟，按下事件立即上报
    printf("\033[?1003l\033[?1002h\033[?1006h");
    fflush(stdout);
    return 0;
}

int32_t touch_read(int32_t *x, int32_t *y, int32_t *is_pressed) {
    if (s_pressed) {
        if (x) *x = s_x;
        if (y) *y = s_y;
        if (is_pressed) *is_pressed = 1;
    }
    else {
        if (is_pressed) *is_pressed = 0;
    }
    return 0;
}
