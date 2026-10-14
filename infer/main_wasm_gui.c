// main_wasm_gui.c —— Nano-WASM-GUI 主入口（Emscripten）
//
// 事件获取与分发：参照 main.c 的拆分事件环路（NANO_UI_SPLIT_EVENT_LOOP，移植自
// linglong_m5core2.ino 的 Core1 生产→Core0 消费模型），在单线程 WASM 下的等价实现：
//
//   生产者（wasm_gui_input_tick）：由页面 JS 驱动——触屏/键盘事件回调（携带
//     performance.timeOrigin+e.timeStamp 的真实时刻）到达即调用 + 50ms 心跳
//     （补充无事件时的持续状态判定，如长按到期）。tick 内逐事件回放
//     get_input_event 采样（每个事件以其真实时刻作为采样时刻），按键/触屏边沿
//     事件按 ino 语义投递到结果事件队列（短按 -1 与触屏边沿可靠投递、长按重复
//     -2 可丢弃）。输入采样与渲染帧率解耦：玲珑仪等秒级渲染场景下，一次 100ms
//     点按不会被误判为长按连发（此前串行环以渲染帧时刻计时所致）。
//
//   消费者（wasm_gui_frame，emscripten_set_main_loop 注册，rAF 节奏）：
//     每帧排空结果事件队列（触屏边沿位掩码并入——同帧 DOWN+UP 叠加为 3，亚帧
//     短点按不湮灭；按键事件每帧至多一个），触屏电平/坐标取生产者高频锁存的
//     共享快照覆盖 key_event，随后 main_event_handler/main_periodic_task。
//     帧回调体对齐 main.c 拆分环路消费者的语义。
//
//   单线程说明：生产/消费均在浏览器主线程（JS 事件循环）内交错执行，无并发；
//   SPSC 队列保留结构语义，无需内存屏障。ino 拆分环路的「玲珑仪/时光集忙提示」
//   （生产端即时 gfx_draw_busy）在 WASM 下省略——渲染阻塞时 JS 回调同样延后，
//   忙提示无法早于长帧上屏，无意义。

#include <locale.h>
#include <stdlib.h>
#include <time.h>

#include <emscripten.h>

#include "platform.h"
#include "ui_app.h"
#include "graphics.h"
#include "hal_key.h" // NANO_KEY_IDLE（队列事件判别用）

// 全局UI状态（与 main.c 同）
static Global_State *global_state = NULL;
static Key_Event    *key_event = NULL;

// ===============================================================================
// 结果事件队列（对齐 main.c 拆分环路：容量 8；生产者仅 tick、消费者仅帧回调，
// 单线程故 head/tail 无需原子屏障）
// ===============================================================================
#define NANO_EVT_QUEUE_CAP (8)

static Key_Event s_evt_buf[NANO_EVT_QUEUE_CAP];
static uint32_t  s_evt_head = 0; // 消费序号（仅消费者写）
static uint32_t  s_evt_tail = 0; // 生产序号（仅生产者写）

// 入队。reliable=1（短按 -1 / 触屏边沿）：满时打印告警并丢弃（对齐 ino 的告警语义；
// 渲染阻塞期间 tick 同样延后，恢复后集中回放可能填满——容量 8 对真实按键速率足够）；
// reliable=0（-2 长按重复流）：满即静默丢弃，消费端每帧取最新，重复节奏对齐帧率。
static int32_t wasm_evtq_push(const Key_Event *ev, int32_t reliable) {
    if (s_evt_tail - s_evt_head >= NANO_EVT_QUEUE_CAP) {
        if (reliable) printf("WARNING: wasm event queue full, reliable event dropped!\n");
        return -1;
    }
    s_evt_buf[s_evt_tail % NANO_EVT_QUEUE_CAP] = *ev;
    s_evt_tail++;
    return 0;
}

// 出队。返回 1 取到，0 为空。
static int32_t wasm_evtq_pop(Key_Event *ev) {
    if (s_evt_head == s_evt_tail) return 0;
    *ev = s_evt_buf[s_evt_head % NANO_EVT_QUEUE_CAP];
    s_evt_head++;
    return 1;
}

// ===============================================================================
// 生产者：输入采样 tick（JS 事件回调/心跳驱动；ts>0 为事件真实时刻（Epoch ms，
// double 传参保精度），ts=0 为心跳（取当前真实时刻））
// ===============================================================================

// 触屏 HAL 的事件队列接口（hal_touch_wasm.c）
extern int32_t wasm_touch_pop(int32_t *action, int32_t *x, int32_t *y, double *ts);
extern void    wasm_touch_set_level(int32_t action, int32_t x, int32_t y);

// 忙提示刚上屏标记（dispatch 置位，frame 消费）：见 wasm_dispatch_input_events 注释
static int32_t s_busy_just_drawn = 0;

// 一次采样结果的投递（对齐 main.c 生产者的投递语义）
static void wasm_dispatch_input_events(Key_Event *ev_prod) {
    // 触屏 DOWN/UP 边沿：可靠投递。必须先于按键事件投递（同一次松手会同时产生
    // 触屏 UP 与宫格软按键下降沿；先投触屏可保证同帧内热点判定先于按键处理）。
    if (ev_prod->touch_edge != 0) {
        Key_Event touch_ev = *ev_prod;
        touch_ev.key_code = NANO_KEY_IDLE;
        touch_ev.key_edge = 0;
        wasm_evtq_push(&touch_ev, 1);
    }
    // 按键事件：短按下降沿(-1)可靠投递；长按/重复动作(-2)可丢弃投递
    if (ev_prod->key_code != NANO_KEY_IDLE && ev_prod->key_edge < 0) {
        // 玲珑仪/时光集：生产端即时忙提示（对齐 linglong_m5core2.ino 与 main.c 拆分
        // 环路：按键事件入队前先把「正在计算」画进帧缓冲并 gfx_refresh 推帧——
        // hal_display_wasm 更新 RGBA side buffer 并 epoch++，页面渲染循环下一 rAF
        // 即上屏，用户在长帧渲染开始前立即看到反馈；业务下一帧渲染先清屏重绘，
        // 提示自然消失，不残留。gfx_draw_busy 是图形层既有设施（graphics.c），
        // 此处仅调用，不改动任何平台无关业务模块。）
        if (global_state->STATE == STATE_LINGLONG || global_state->STATE == STATE_ALBUM) {
            gfx_draw_busy(global_state->gfx);
            gfx_refresh(global_state->gfx);
            // 标记「忙提示刚上屏」：消费者帧回调本帧跳过（不消费不渲染），
            // 让浏览器先把提示 paint 出来——浏览器渲染流水线中 rAF 回调先于 paint，
            // 若本帧照常渲染，提示会在 paint 之前被覆盖，永远到不了屏幕（慢设备实测不可见）。
            s_busy_just_drawn = 1;
        }
        wasm_evtq_push(ev_prod, (ev_prod->key_edge == -1) ? 1 : 0);
    }
}

EMSCRIPTEN_KEEPALIVE
void wasm_gui_input_tick(double ts) {
    // 事件采样状态机（get_input_event）依赖 key_event 的持久状态（prev_key/
    // key_timer/key_mask/key_repeat/触摸边沿锁存等跨调用保持）——必须为静态持久
    //（对齐 main.c 拆分环路生产者的 `static Key_Event ev_prod`）；
    // 每次清零会把 prev_key 复位为 IDLE，导致上升沿每 tick 重复触发、状态机崩坏。
    static Key_Event ev_prod;
    int32_t replayed = 0;

    // 逐事件回放触摸队列：每个事件以其真实时刻驱动一次采样（电平先回放）
    int32_t action, x, y;
    double ev_ts;
    while (wasm_touch_pop(&action, &x, &y, &ev_ts)) {
        wasm_touch_set_level(action, x, y);
        global_state->timestamp = (uint64_t)ev_ts; // 采样时刻=事件真实时刻
        get_input_event(&ev_prod, global_state);
        wasm_dispatch_input_events(&ev_prod);
        replayed = 1;
    }

    // 心跳/键盘事件后：以真实时刻采样当前电平（键盘电平由 hal_key_wasm 维护，
    // 键盘事件回调在置位电平后立即调本函数——其边沿即以事件真实时刻计时）；
    // 无事件时本 tick 承担长按到期等持续状态判定。
    if (!replayed) {
        global_state->timestamp = (ts > 0.0) ? (uint64_t)ts : get_timestamp_in_ms();
        get_input_event(&ev_prod, global_state);
        wasm_dispatch_input_events(&ev_prod);
    }

    // 更新上一轮循环的物理时间戳（对齐 ino 生产者侧语义）
    global_state->timestamp_last = global_state->timestamp;
}

// ===============================================================================
// 消费者：每帧回调（emscripten 主循环，rAF 节奏；对齐 main.c 拆分环路消费者）
// ===============================================================================

// 调试导出：当前 UI 状态机状态（STATE_*，ui_app.h）。供端到端测试脚本观测状态轨迹。
EMSCRIPTEN_KEEPALIVE
int32_t wasm_gui_get_state(void) {
    return global_state ? global_state->STATE : -9999;
}

static void wasm_gui_frame(void) {
    // 忙提示刚上屏：本帧跳过（不消费不渲染），让浏览器先把提示 paint 出来
    if (s_busy_just_drawn) {
        s_busy_just_drawn = 0;
        return;
    }
    // 排空事件队列：触屏边沿事件位掩码并入本帧（同帧 DOWN+UP 叠加为 3），
    // 坐标/电平取事件内最新值；按键事件取出首个即停（每帧至多一个）
    key_event->touch_edge = 0;
    uint8_t frame_touch_edge = 0;
    {
        Key_Event ev;
        int32_t got_key = 0;
        while (wasm_evtq_pop(&ev)) {
            if (ev.key_code == NANO_KEY_IDLE && ev.touch_edge != 0) {
                frame_touch_edge |= ev.touch_edge;
                key_event->touch_x = ev.touch_x;
                key_event->touch_y = ev.touch_y;
                key_event->is_touching = ev.is_touching;
                key_event->touch_down_x = ev.touch_down_x;
                key_event->touch_down_y = ev.touch_down_y;
                continue;
            }
            *key_event = ev; // 按键事件：整体沿用（其触屏字段亦为生产端最新值）
            got_key = 1;
            break;
        }
        if (!got_key) {
            key_event->key_code = NANO_KEY_IDLE;
            key_event->key_edge = 0;
        }
    }
    key_event->touch_edge = frame_touch_edge;

    // 触屏电平/坐标取生产端高频锁存的共享快照；touch_edge/touch_down_* 不被覆盖
    key_event->touch_x = global_state->touch_x;
    key_event->touch_y = global_state->touch_y;
    key_event->is_touching = global_state->is_touching;

    // 事件处理器
    main_event_handler(key_event, global_state);
    // 周期性任务
    main_periodic_task(key_event, global_state);
}

int main() {
    if (!setlocale(LC_CTYPE, "")) return -1;

    key_event = (Key_Event*)platform_calloc(1, sizeof(Key_Event));
    global_state = (Global_State*)platform_calloc(1, sizeof(Global_State));

    main_init(key_event, global_state);

    // rAF 节奏主循环（simulate_infinite_loop=1：不返回，runtime 不退出）
    emscripten_set_main_loop(wasm_gui_frame, 0, 1);

    // 不会到达（emscripten_set_main_loop 以 longjmp 模拟无限循环）
    main_deinit(key_event, global_state);
    free(global_state);
    free(key_event);

    return 0;
}
