////////////////////////////////////////////////////////////////////////////
//  Native Library : Key —— 设计说明见 am_native_Key.h
//
//  数据源：nano_key_events（HAL 投递的边沿事件旁路队列），不是
//  input_device_read_key()——后者是 GUI 输入层的消费点，Scheme 侧读它就会
//  与 GUI 抢事件（详见 nano_key_events.h 的说明）。
//
//  本文件做三件事：
//    1) 每次进入任一过程都先把 HAL 旁路队列里的边沿**排空**到本库自己的两个
//       事件栈（按下栈 / 松开栈），于是 take_press 与 take_release 互不吞噬；
//    2) is_down 直接问 HAL 的电平表（它由投递侧维护，与队列消费无关）；
//    3) 结果按单值栈纪律压回操作数栈（N1/N6），单一出口。
////////////////////////////////////////////////////////////////////////////

#include <stdbool.h>
#include <stdint.h>

#include "am_native_Key.h"
#include "nano_key_events.h"

// 本库自己的事件栈（由排空 HAL 队列得到）。深度比 HAL 队列大一倍：
// HAL 队列一次最多积压 32 条边沿，排空后可能有 32 条同向事件待 Scheme 取走。
#define KEY_LOCAL_QUEUE_LEN (64)

static uint8_t s_press_q[KEY_LOCAL_QUEUE_LEN];
static uint32_t s_press_head = 0, s_press_count = 0;

static uint8_t s_release_q[KEY_LOCAL_QUEUE_LEN];
static uint32_t s_release_head = 0, s_release_count = 0;

static uint32_t s_local_dropped = 0; // 本库事件栈溢出（Scheme 侧长时间不取）

// 最近一次被取走的按下沿（键码 + 宿主时刻），供模态输入框剔除"打开它的那次按键"的松开沿
static uint8_t  s_last_press_code = 0;
static uint64_t s_last_press_ms = 0;

// 把 HAL 旁路队列里的所有边沿事件排空到本地两个事件栈。
static void key_drain_hal(void) {
    uint8_t code = 0, down = 0;
    while (nano_key_events_pop(&code, &down)) {
        uint8_t *q = down ? s_press_q : s_release_q;
        uint32_t *head = down ? &s_press_head : &s_release_head;
        uint32_t *count = down ? &s_press_count : &s_release_count;

        if (*count >= KEY_LOCAL_QUEUE_LEN) { // 栈满：丢最旧（与 HAL 队列同策略）
            *head = (*head + 1) % KEY_LOCAL_QUEUE_LEN;
            (*count)--;
            s_local_dropped++;
        }
        q[(*head + *count) % KEY_LOCAL_QUEUE_LEN] = code;
        (*count)++;
    }
}

// 从指定事件栈取出最早一个键码；空栈返回 0（NANO_KEY_IDLE）。
static uint8_t key_take(uint8_t *q, uint32_t *head, uint32_t *count) {
    uint8_t code;
    if (*count == 0) return 0;
    code = q[*head];
    *head = (*head + 1) % KEY_LOCAL_QUEUE_LEN;
    (*count)--;
    return code;
}

void am_native_Key_last_press(uint8_t *code, uint64_t *timestamp_ms) {
    if (code) *code = s_last_press_code;
    if (timestamp_ms) *timestamp_ms = s_last_press_ms;
}

// ---------------------------------------------------------------------------
// 值型收尾（契约 N1 + N6）：压 1 个值 + step 1 次。全部过程共用单一出口。
// ---------------------------------------------------------------------------
static int32_t key_push(am_process_t *proc, am_value_t v) {
    if (am_process_push_operand(proc, v) != 0) return -1;
    am_process_step(proc);
    return 0;
}

static int32_t key_push_bool(am_process_t *proc, bool b) {
    return key_push(proc, b ? AM_VALUE_TRUE : AM_VALUE_FALSE);
}

static int32_t key_push_int(am_process_t *proc, int32_t n) {
    return key_push(proc, am_make_value_of_int((am_int_t)n));
}

// 从操作数栈弹一个数值键码（越界/非数值一律按 NANO_KEY_IDLE 处理）。
static uint8_t key_pop_code(am_process_t *proc) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return 0;
    if (am_value_is_float(v)) {
        am_float_t f = am_value_to_float(v);
        if (!(f > 0.0f) || !(f < 256.0f)) return 0;
        return (uint8_t)f;
    }
    if (am_value_is_int(v)) {
        am_int_t i = am_value_to_int(v);
        return (i > 0 && i < 256) ? (uint8_t)i : 0;
    }
    if (am_value_is_uint(v)) {
        am_uint_t u = am_value_to_uint(v);
        return (u > 0 && u < 256) ? (uint8_t)u : 0;
    }
    return 0;
}

// ===============================================================================
// Native 函数实现
// ===============================================================================

// 清空事件源与本地事件栈（宿主在启动 App 时调用，见头文件）
void am_native_Key_reset(void) {
    nano_key_events_reset();
    s_last_press_code = 0;
    s_last_press_ms = 0;
    s_press_head = 0;
    s_press_count = 0;
    s_release_head = 0;
    s_release_count = 0;
    s_local_dropped = 0;
}

int32_t am_native_Key_is_available(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    key_drain_hal(); // 顺手排空，保证「可用」与「事件已就位」一致
    return key_push_bool(proc, nano_key_events_has_source() == 1);
}

int32_t am_native_Key_is_down(am_runtime_t *rt, am_process_t *proc) {
    uint8_t code;
    (void)rt;
    code = key_pop_code(proc);
    key_drain_hal(); // 排空以免事件堆积（电平表由投递侧维护，与此无关）
    return key_push_bool(proc, code != 0 && nano_key_events_is_down(code) == 1);
}

int32_t am_native_Key_take_press(am_runtime_t *rt, am_process_t *proc) {
    uint8_t code;
    key_drain_hal();
    code = key_take(s_press_q, &s_press_head, &s_press_count);
    if (code != 0) {
        // 用虚拟机时钟（各宿主经 vtable 注入）而不是平台时间函数：与宿主时间线一致，
        // 且回归台的虚拟时钟下可复现（模态框剔残留松开沿的判据就靠这个时刻）
        s_last_press_code = code;
        s_last_press_ms = (uint64_t)am_runtime_now_ms(rt);
    }
    return key_push_int(proc, (int32_t)code);
}

int32_t am_native_Key_take_release(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    key_drain_hal();
    return key_push_int(proc, (int32_t)key_take(s_release_q, &s_release_head, &s_release_count));
}

int32_t am_native_Key_dropped(am_runtime_t *rt, am_process_t *proc) {
    uint32_t total;
    (void)rt;
    key_drain_hal();
    total = nano_key_events_dropped() + s_local_dropped;
    return key_push_int(proc, (int32_t)total);
}

// ===============================================================================
// 库注册表
// ===============================================================================

static const am_native_func_entry_t am_native_Key_funcs[] = {
    { L"is_available",  am_native_Key_is_available },
    { L"is_down",       am_native_Key_is_down },
    { L"take_press",    am_native_Key_take_press },
    { L"take_release",  am_native_Key_take_release },
    { L"dropped",       am_native_Key_dropped },
};

const am_native_lib_entry_t am_native_Key_lib = {
    L"Key",
    am_native_Key_funcs,
    sizeof(am_native_Key_funcs) / sizeof(am_native_Key_funcs[0])
};