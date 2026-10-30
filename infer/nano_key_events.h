#ifndef __NANO_KEY_EVENTS_H__
#define __NANO_KEY_EVENTS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

////////////////////////////////////////////////////////////////////////////
//  nano_key_events —— 按键事件旁路队列（平台无关）
//
//  为什么需要它：
//    hal_key.h 的 input_device_read_key() 是「电平 + 单槽位」契约——GUI 输入层
//    （ui_app.c 的 get_input_event）每帧读一次，用于边沿判定、长按连发与 4x4
//    宫格映射。它有两个性质使它不适合直接给「灵机应用程序」（Scheme App）用：
//      1) 读即消费：wasm HAL 读到电平后会清掉「未读锁存」，ncurses HAL 更是直接
//         从 stdin 取走字节——Scheme 侧再读一次就与 GUI 抢事件（谁先读谁拿到，
//         两边的行为都变得不确定）；
//      2) 只有电平：一次「按下→松开」的完整事件流（含亚帧点按）无法可靠重建。
//    本模块因此是一条【旁路】：HAL 在注入按键的同时把边沿事件投递进来，GUI 的
//    读取路径完全不受影响（与 am_native_Touch 的「只读旁路」同一哲学），Scheme
//    侧经 (native Key) 库消费（见 am_native_Key.h）。
//
//  设计约定：
//    · 只投递【边沿】——push 内部按每键电平表去重，按住期间的重复 keydown
//      （浏览器 auto-repeat）不会产生新事件；is_down 由同一张电平表回答。
//    · 队列满时丢弃最旧事件并计数（nano_key_events_dropped），绝不阻塞投递方
//      （投递发生在输入回调/中断式上下文里，不能等）。
//    · 无锁：当前接入的两个平台（WASM GUI 单线程、终端 ncurses 单线程）都在
//      同一线程内投递与消费。将来若接入 ESP32 的独立按键扫描任务，须由该任务
//      投递、主循环消费，并在此处补临界区。
////////////////////////////////////////////////////////////////////////////

// 队列深度（容纳的边沿事件条数）。一帧内人手最多几个事件，32 有充足余量。
#define NANO_KEY_EVENT_QUEUE_LEN (32)

// 声明「本平台的键盘 HAL 会投递事件」（由 HAL 的 input_device_init 调用）。
// 未声明的平台上 (Key.is_available) 返回 #f——Scheme App 据此降级为纯触屏交互。
void nano_key_events_mark_source(void);

// 本平台是否已声明事件源（1-是，0-否）
int32_t nano_key_events_has_source(void);

// 投递一个按键边沿。key_code 为 hal_key.h 的 NANO_KEY_* 逻辑键值
// （NANO_KEY_IDLE 与超出 0..255 的值一律忽略）；is_down：1-按下，0-松开。
// 调用方无需去重：电平未变化时本函数自行丢弃。
void nano_key_events_push(uint8_t key_code, uint8_t is_down);

// 取走一个最早的边沿事件。返回 1 表示取到（*key_code/*is_down 已填），0 表示队列空。
int32_t nano_key_events_pop(uint8_t *key_code, uint8_t *is_down);

// 该键当前是否按住（由投递侧维护的电平表回答，与队列是否已被消费无关）。
int32_t nano_key_events_is_down(uint8_t key_code);

// 因队列满而被丢弃的事件累计条数（诊断用；0 表示从未丢事件）。
uint32_t nano_key_events_dropped(void);

// 清空队列与电平表（回到「没有任何键按下」的初始态）。
void nano_key_events_reset(void);

#ifdef __cplusplus
}
#endif

#endif