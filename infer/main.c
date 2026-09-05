#include <locale.h>
#include <stdlib.h>
#include <time.h>
#include "platform.h"
#include "ui_app.h"
#include "graphics.h"
#include "hal_key.h" // NANO_KEY_IDLE（拆分事件环路的队列事件判别用）

// 全局UI状态

static Global_State *global_state = NULL;
static Key_Event    *key_event = NULL;

// 进程时区：Luckfox-Pico-86-Panel 的 buildroot 镜像无时区配置（系统时间为 UTC），
// 业务期望中国标准时间（东八区）。POSIX TZ 串 "CST-8" 由 libc 直接解析，无需 tzdata。
//（与 linglong_m5core2.ino 中 setenv("TZ","CST-8") 同一约定；仅本机型定义）
static void nano_init_timezone(void) {
#if defined(NANO_POD_LUCKFOX)
    setenv("TZ", "CST-8", 1);
    tzset();
#endif
}

#if defined(NANO_UI_SPLIT_EVENT_LOOP)
// ===============================================================================
// 事件生产/消费拆分架构（移植 linglong_m5core2.ino 的 Core1→Core0 模型，宏门控，
// 仅定义 NANO_UI_SPLIT_EVENT_LOOP 的目标启用——当前为 Luckfox-Pico-86-Panel）：
//   生产者线程：1~2ms 高频调用 get_input_event（零改动，其设计本为生产端），
//               触屏 DOWN/UP 边沿与短按(-1)可靠投递、长按重复(-2)可丢弃投递，
//               触屏电平/坐标经 global_state 共享快照传递（不入队）；
//   消费者（主循环）：每帧排空队列——触屏边沿位掩码并入（同帧 DOWN+UP 叠加为 3，
//               亚帧短点按不湮灭），坐标/电平取快照，按键事件每帧至多一个，
//               随后照常 main_event_handler/main_periodic_task。
// 解决：玲珑仪等秒级渲染场景下，串行环每帧才采样一次输入导致短按丢失、
//       按住连发的问题（业务层语义与 ESP32 参考实现完全一致，见 ui.h 注释）。
// ===============================================================================

// SPSC 无锁事件队列（仅生产者写 tail、仅消费者写 head；容量同 .ino 的 8）
#define NANO_EVT_QUEUE_CAP (8)
typedef struct {
    Key_Event buf[NANO_EVT_QUEUE_CAP];
    volatile uint32_t head; // 消费序号（单调，仅消费者写）
    volatile uint32_t tail; // 生产序号（单调，仅生产者写）
} Nano_Event_Queue;

static Nano_Event_Queue s_evt_queue;

// 入队。reliable=1：满时 1ms 后重试一次（对齐 .ino 的 1ms 超时+告警语义，一次性
// 事件不可丢）；reliable=0（-2 长按重复流）：满即静默丢弃，消费端每帧取最新，
// 重复节奏自然对齐帧率。返回 0 成功。
static int32_t nano_evtq_push(const Key_Event *ev, int32_t reliable) {
    for (int32_t tries = 0; tries < 2; tries++) {
        uint32_t t = s_evt_queue.tail, h = s_evt_queue.head;
        if (t - h < NANO_EVT_QUEUE_CAP) {
            s_evt_queue.buf[t % NANO_EVT_QUEUE_CAP] = *ev;
            platform_memory_barrier();   // 数据先于写序号发布
            s_evt_queue.tail = t + 1;
            return 0;
        }
        if (!reliable) return -1;
        platform_task_delay_ms(1);
    }
    printf("WARNING: event queue send timeout (reliable event dropped)!\n");
    return -1;
}

// 出队。返回 1 取到，0 为空。
static int32_t nano_evtq_pop(Key_Event *ev) {
    uint32_t h = s_evt_queue.head, t = s_evt_queue.tail;
    platform_memory_barrier(); // 读序号先于数据读取
    if (h == t) return 0;
    *ev = s_evt_queue.buf[h % NANO_EVT_QUEUE_CAP];
    s_evt_queue.head = h + 1;
    return 1;
}

// 生产者任务（对应 .ino 的 Core1 loop）：高频采样输入并投递事件
static void nano_input_producer_task(void *arg) {
    (void)arg;
    Key_Event ev_prod = {0};
    while (1) {
        // 物理时间戳（时间戳生产/消费关系同 .ino：生产端更新，消费端只读）
        global_state->timestamp = get_timestamp_in_ms();

        // 获取输入事件（按键 + 触屏）：触屏电平快照由本调用写入 global_state
        get_input_event(&ev_prod, global_state);

        // 触屏 DOWN/UP 边沿事件：可靠投递。必须先于按键事件投递（同一次松手会同时
        // 产生触屏 UP 与宫格软按键下降沿；先投触屏可保证同帧内热点判定先于按键处理）。
        // 本平台无电源键（NANO_HAS_PWR_KEY=0），锁屏抑制逻辑不参与。
        if (ev_prod.touch_edge != 0) {
            Key_Event touch_ev = ev_prod;
            touch_ev.key_code = NANO_KEY_IDLE;
            touch_ev.key_edge = 0;
            nano_evtq_push(&touch_ev, 1);
        }

        // 按键事件：短按下降沿(-1)可靠投递；长按/重复动作(-2)可丢弃投递
        if (ev_prod.key_code != NANO_KEY_IDLE && ev_prod.key_edge < 0) {
            int32_t reliable = (ev_prod.key_edge == -1);
            // 玲珑仪/时光集：生产端即时忙提示（忠实移植 .ino；与消费端并发访问 gfx
            // 为 cosmetic 级风险，最坏画面撕裂、下帧自愈——ESP32 端同样如此）
            if (global_state->STATE == STATE_LINGLONG || global_state->STATE == STATE_ALBUM) {
                gfx_draw_busy(global_state->gfx);
                gfx_refresh(global_state->gfx);
            }
            nano_evtq_push(&ev_prod, reliable);
        }

        // 更新上一轮循环的物理时间戳
        global_state->timestamp_last = global_state->timestamp;

        platform_task_delay_ms(1);
    }
}

int main() {
    nano_init_timezone();
    if(!setlocale(LC_CTYPE, "")) return -1;

    key_event = (Key_Event*)platform_calloc(1, sizeof(Key_Event));
    global_state = (Global_State*)platform_calloc(1, sizeof(Global_State));

    main_init(key_event, global_state);

    // 启动输入生产者任务（栈 4KB：仅输入采样与队列操作，无递归求值）
    platform_task_handle_t input_task = NULL;
    if (platform_task_create(nano_input_producer_task, "input", 4096,
                             NULL, 1, -1, &input_task) != 0) {
        printf("FATAL: input producer task create failed\n");
        return -1;
    }

    while (1) {
        // 排空事件队列（语义同 .ino core0_render_task）：
        // 触屏边沿事件位掩码并入本帧，按键事件取出首个即停（每帧至多一个）
        key_event->touch_edge = 0;
        uint8_t frame_touch_edge = 0;
        {
            Key_Event ev;
            int32_t got_key = 0;
            while (nano_evtq_pop(&ev)) {
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

        // 仅玲珑仪/时光集显示忙提示（消费端侧，同 .ino）
        if (key_event->key_code != NANO_KEY_IDLE &&
            (global_state->STATE == STATE_LINGLONG || global_state->STATE == STATE_ALBUM)) {
            gfx_draw_busy(global_state->gfx);
            gfx_refresh(global_state->gfx);
        }

        // 事件处理器
        main_event_handler(key_event, global_state);
        // 周期性任务
        main_periodic_task(key_event, global_state);
    }

    main_deinit(key_event, global_state);
    free(global_state);
    free(key_event);

    return 0;
}

#else // !NANO_UI_SPLIT_EVENT_LOOP：串行单循环（原始路径，其余目标不变）

int main() {
    nano_init_timezone();
    if(!setlocale(LC_CTYPE, "")) return -1;

    key_event = (Key_Event*)platform_calloc(1, sizeof(Key_Event));
    global_state = (Global_State*)platform_calloc(1, sizeof(Global_State));

    main_init(key_event, global_state);

    while (1) {
        // 物理时间戳
        global_state->timestamp = get_timestamp_in_ms();
        // 获取输入事件（按键 + 触屏）
        get_input_event(key_event, global_state);
        // 事件处理器
        main_event_handler(key_event, global_state);
        // 周期性任务
        main_periodic_task(key_event, global_state);
        // 更新上一轮循环的物理时间戳
        global_state->timestamp_last = global_state->timestamp;
    }

    main_deinit(key_event, global_state);
    free(global_state);
    free(key_event);

    return 0;
}
#endif
