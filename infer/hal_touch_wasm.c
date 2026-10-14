// hal_touch_wasm.c —— Nano-WASM-GUI 触屏 HAL（Emscripten）
//
// 触屏事件来源：页面 JS 的 Pointer Events（pointerdown/move/up，鼠标/触屏统一）。
//
// 事件队列模型（2026-10-14 重写，修复慢渲染场景点按误判长按连发）：
// 参照 main.c 拆分事件环路（移植自 linglong_m5core2.ino）的生产者/消费者语义——
// 输入采样必须在【真实物理节奏】上进行，边沿/长短按判定以【事件发生的真实时刻】
// 为准，与渲染帧率解耦。此前 JS 注入仅维护电平+点按锁存，由渲染帧里的
// get_input_event 轮询：玲珑仪等秒级渲染场景下，一次 100ms 点按的 DOWN/UP 沿被拆到
// 相隔数秒的帧中，按帧时刻计时的「按住时长」超 360ms 阈值 → 误判长按并连发 -2，
// 表现为按一下触发多次动作。
//
// 本 HAL 将 JS 触屏事件（带 performance.timeOrigin+e.timeStamp 的真实时刻）入队；
// 由 main_wasm_gui.c 的生产者 tick（wasm_gui_input_tick）逐事件回放驱动
// get_input_event 采样（每个事件以其真实时刻作为采样时刻），边沿/长短按判定
// 恢复物理正确。move 事件入队时可合并（队尾为 move 则覆盖，拖动轨迹不丢失精度
// 诉求之外的高频冗余）；DOWN/UP 为可靠事件，满时告警丢弃（容量 16 足够）。

#include "platform.h"
#include "hal_touch.h"

#include <emscripten.h>

#define NANO_TOUCH_EVT_CAP (16)

typedef struct {
    int32_t action;   // 0=松开 1=按下 2=移动
    int32_t x, y;     // 帧缓冲像素坐标
    double  ts;       // 事件真实时刻（Epoch 毫秒；double 53 位精度装 Epoch 毫秒安全）
} Wasm_Touch_Evt;

static Wasm_Touch_Evt s_evtq[NANO_TOUCH_EVT_CAP];
static volatile uint32_t s_evt_head = 0, s_evt_tail = 0; // 单线程（浏览器主线程），无需屏障

// 当前触屏电平（由生产者 tick 逐事件回放时更新；touch_read 读取）
static int32_t s_pressed = 0;
static int32_t s_x = 0;
static int32_t s_y = 0;

// 触屏事件入口（页面 JS ccall）：ts 为事件真实时刻（Epoch 毫秒，double）
EMSCRIPTEN_KEEPALIVE
void wasm_gui_on_touch(int32_t action, int32_t x, int32_t y, double ts) {
    // move 合并：队尾为 move 则原地更新（拖动高频轨迹点可丢弃，取最新）
    if (action == 2 && s_evt_tail != s_evt_head) {
        Wasm_Touch_Evt *last = &s_evtq[(s_evt_tail - 1) % NANO_TOUCH_EVT_CAP];
        if (last->action == 2) {
            last->x = x;
            last->y = y;
            last->ts = ts;
            return;
        }
    }
    if (s_evt_tail - s_evt_head >= NANO_TOUCH_EVT_CAP) {
        if (action != 2) printf("WARNING: wasm touch event queue full, edge event dropped!\n");
        return; // move 静默丢弃
    }
    s_evtq[s_evt_tail % NANO_TOUCH_EVT_CAP].action = action;
    s_evtq[s_evt_tail % NANO_TOUCH_EVT_CAP].x = x;
    s_evtq[s_evt_tail % NANO_TOUCH_EVT_CAP].y = y;
    s_evtq[s_evt_tail % NANO_TOUCH_EVT_CAP].ts = ts;
    s_evt_tail++;
}

// 出队（生产者 tick 调用）。返回 1 取到，0 为空。
int32_t wasm_touch_pop(int32_t *action, int32_t *x, int32_t *y, double *ts) {
    if (s_evt_head == s_evt_tail) return 0;
    const Wasm_Touch_Evt *ev = &s_evtq[s_evt_head % NANO_TOUCH_EVT_CAP];
    *action = ev->action;
    *x = ev->x;
    *y = ev->y;
    *ts = ev->ts;
    s_evt_head++;
    return 1;
}

// 回放一个事件到电平（生产者 tick 在调用 get_input_event 前调用）
void wasm_touch_set_level(int32_t action, int32_t x, int32_t y) {
    if (action == 1) {          // 按下
        s_pressed = 1;
        s_x = x;
        s_y = y;
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
    }
    else {
        // 未按压不写坐标（上层约定：保持上一轮最后有效值）
        if (is_pressed) *is_pressed = 0;
    }
    return 0;
}
