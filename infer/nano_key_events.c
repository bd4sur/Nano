////////////////////////////////////////////////////////////////////////////
//  nano_key_events —— 按键事件旁路队列（设计说明见 nano_key_events.h）
//
//  实现：一个定长环形队列 + 一张 256 位电平表。
//    · push 先查电平表：电平没变则直接返回（浏览器 auto-repeat 因此天然去抖）；
//    · 电平变了才入队，队列满则丢弃最旧一条并计数（投递方永不阻塞）；
//    · is_down 直接查电平表——它与「队列是否已被消费」解耦，所以 Scheme 侧
//      即使一帧不取事件，(Key.is_down n) 依然是当拍的真实电平。
////////////////////////////////////////////////////////////////////////////

#include <stddef.h>

#include "nano_key_events.h"

// 边沿事件（8 位键码 + 1 位方向）
typedef struct {
    uint8_t key_code;
    uint8_t is_down;
} nano_key_event_t;

static nano_key_event_t s_queue[NANO_KEY_EVENT_QUEUE_LEN];
static uint32_t s_head = 0;   // 读游标（最旧事件）
static uint32_t s_count = 0;  // 当前队列内事件条数
static uint32_t s_dropped = 0;

static int32_t s_has_source = 0;

// 每键电平表（256 位 = 32 字节）
static uint8_t s_down_bits[32];

static int key_is_down(uint8_t k) {
    return (s_down_bits[k >> 3] >> (k & 7)) & 1;
}

static void key_set_down(uint8_t k, int down) {
    if (down) s_down_bits[k >> 3] |= (uint8_t)(1u << (k & 7));
    else      s_down_bits[k >> 3] &= (uint8_t)~(1u << (k & 7));
}

void nano_key_events_mark_source(void) {
    s_has_source = 1;
}

int32_t nano_key_events_has_source(void) {
    return s_has_source;
}

void nano_key_events_push(uint8_t key_code, uint8_t is_down) {
    if (key_code == 0) return; // NANO_KEY_IDLE：不是有效键码

    int want_down = is_down ? 1 : 0;
    if (key_is_down(key_code) == want_down) return; // 电平未变：去重（auto-repeat 无害）
    key_set_down(key_code, want_down);

    if (s_count >= NANO_KEY_EVENT_QUEUE_LEN) {
        // 队列满：丢弃最旧一条（前进读游标），保证最新事件一定进得来
        s_head = (s_head + 1) % NANO_KEY_EVENT_QUEUE_LEN;
        s_count--;
        s_dropped++;
    }

    uint32_t tail = (s_head + s_count) % NANO_KEY_EVENT_QUEUE_LEN;
    s_queue[tail].key_code = key_code;
    s_queue[tail].is_down = (uint8_t)want_down;
    s_count++;
}

int32_t nano_key_events_pop(uint8_t *key_code, uint8_t *is_down) {
    if (s_count == 0) return 0;

    nano_key_event_t ev = s_queue[s_head];
    s_head = (s_head + 1) % NANO_KEY_EVENT_QUEUE_LEN;
    s_count--;

    if (key_code) *key_code = ev.key_code;
    if (is_down) *is_down = ev.is_down;
    return 1;
}

int32_t nano_key_events_is_down(uint8_t key_code) {
    return key_is_down(key_code);
}

uint32_t nano_key_events_dropped(void) {
    return s_dropped;
}

void nano_key_events_reset(void) {
    s_head = 0;
    s_count = 0;
    s_dropped = 0;
    for (size_t i = 0; i < sizeof s_down_bits; i++) s_down_bits[i] = 0;
}