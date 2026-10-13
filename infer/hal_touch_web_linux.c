// hal_touch_web_linux.c —— Nano-Web 触屏 HAL
//
// 触屏事件来源：网页客户端经 WebSocket 上报（nano_ws_server 的客户端接收线程
// 调用本文件的 web_remote_on_touch），语义对齐物理触屏的电平/坐标轮询模型。
//
// 点按保底锁存（s_down_pending）：按下事件到达后，至少被 touch_read 以
// pressed=1 消费一次才清除——否则一次快速点按（DOWN+UP 落在主循环两次轮询
// 之间）会被整个吞掉。nano_tty 的 ncurses 触屏（hal_touch_ncurses_linux.c）
// 存在同样的亚帧点按丢失（TTY 由拆分事件环路规避，本 HAL 在数据源端解决，
// 串行主循环下同样可靠）。

#include "platform.h"
#include "hal_touch.h"
#include "nano_ws_server.h" // web_remote_on_touch 由本文件实现（ws 服务器接收线程回调）

#include <pthread.h>

// 触屏状态（ws 服务器接收线程写，主循环 touch_read 读）
static pthread_mutex_t s_touch_lock = PTHREAD_MUTEX_INITIALIZER;
static int32_t s_pressed = 0;   // 当前电平：1-触摸中
static int32_t s_x = 0;         // 最近有效触点坐标
static int32_t s_y = 0;
static int32_t s_down_pending = 0; // 1-有未被消费过的按下（点按保底锁存）
static int32_t s_down_x = 0;    // 锁存的按下点坐标
static int32_t s_down_y = 0;

// 触屏事件入口（nano_ws_server 客户端接收线程调用）：
// action 0=松开 1=按下 2=移动；x/y 为帧缓冲像素坐标（客户端已按画面缩放换算）。
void web_remote_on_touch(int32_t action, int32_t x, int32_t y) {
    pthread_mutex_lock(&s_touch_lock);
    if (action == 1) {          // 按下
        s_pressed = 1;
        s_x = x;
        s_y = y;
        s_down_pending = 1;
        s_down_x = x;
        s_down_y = y;
    }
    else if (action == 2) {     // 移动（仅触摸期间更新坐标，语义同物理触屏）
        if (s_pressed) {
            s_x = x;
            s_y = y;
        }
    }
    else {                      // 松开：保持最后有效坐标不写（松开瞬间坐标不保证有效，
        s_pressed = 0;          // 上层 get_input_event 会以其锁存的最后触点代替）
    }
    pthread_mutex_unlock(&s_touch_lock);
}

int32_t touch_init() {
    return 0;
}

int32_t touch_read(int32_t *x, int32_t *y, int32_t *is_pressed) {
    pthread_mutex_lock(&s_touch_lock);
    if (s_pressed) {
        if (x) *x = s_x;
        if (y) *y = s_y;
        if (is_pressed) *is_pressed = 1;
        s_down_pending = 0;
    }
    else if (s_down_pending) {
        // DOWN/UP 落在同一轮询间隔内（快速点按）：补报一次按下（按下点坐标），
        // 上层边沿检测将看到完整的 DOWN→UP 序列，点按不再因轮询粒度丢失
        if (x) *x = s_down_x;
        if (y) *y = s_down_y;
        if (is_pressed) *is_pressed = 1;
        s_down_pending = 0;
    }
    else {
        // 未按压：不写坐标（上层约定：保持上一轮最后有效值）
        if (is_pressed) *is_pressed = 0;
    }
    pthread_mutex_unlock(&s_touch_lock);
    return 0;
}
