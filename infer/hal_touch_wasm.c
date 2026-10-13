// hal_touch_wasm.c —— Nano-WASM-GUI 触屏 HAL（Emscripten）
//
// 触屏事件来源：页面 JS 的 Pointer Events（pointerdown/move/up，鼠标/触屏统一）
// 经 ccall 调用本文件导出的 wasm_gui_on_touch 注入；touch_read 供上层轮询。
// 点按保底锁存与 hal_touch_web_linux.c 同（快速点按 DOWN+UP 落在同一帧间隔
// 时补报一次按下，防亚帧点按丢失）。

#include "platform.h"
#include "hal_touch.h"

#include <emscripten.h>

static int32_t s_pressed = 0;
static int32_t s_x = 0;
static int32_t s_y = 0;
static int32_t s_down_pending = 0;
static int32_t s_down_x = 0;
static int32_t s_down_y = 0;

// 触屏事件入口（页面 JS ccall）：action 0=松开 1=按下 2=移动；x/y 为帧缓冲像素坐标
EMSCRIPTEN_KEEPALIVE
void wasm_gui_on_touch(int32_t action, int32_t x, int32_t y) {
    if (action == 1) {          // 按下
        s_pressed = 1;
        s_x = x;
        s_y = y;
        s_down_pending = 1;
        s_down_x = x;
        s_down_y = y;
    }
    else if (action == 2) {     // 移动（仅触摸期间更新坐标）
        if (s_pressed) {
            s_x = x;
            s_y = y;
        }
    }
    else {                      // 松开：保持最后有效坐标不写
        s_pressed = 0;
    }
}

int32_t touch_init() {
    return 0;
}

int32_t touch_read(int32_t *x, int32_t *y, int32_t *is_pressed) {
    if (s_pressed) {
        if (x) *x = s_x;
        if (y) *y = s_y;
        if (is_pressed) *is_pressed = 1;
        s_down_pending = 0;
    }
    else if (s_down_pending) {
        // DOWN/UP 落在同一轮询间隔（快速点按）：补报一次按下
        if (x) *x = s_down_x;
        if (y) *y = s_down_y;
        if (is_pressed) *is_pressed = 1;
        s_down_pending = 0;
    }
    else {
        if (is_pressed) *is_pressed = 0; // 未按压不写坐标
    }
    return 0;
}
