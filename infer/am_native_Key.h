#ifndef __AM_NATIVE_KEY_H__
#define __AM_NATIVE_KEY_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

////////////////////////////////////////////////////////////////////////////
//  Native Library : Key —— 灵机（Animac）键盘本地库
//
//  为什么需要本库：
//    「灵机应用程序」（Scheme App）此前只有 Touch / Imu 两条输入通道，键盘完全
//    不可见——而 GUI 层早就支持实体键盘（Nano-Pod 的 16 键矩阵、M5 的 Faces
//    Keyboard、终端键盘）与浏览器的物理键盘。于是「可打字」的 App（便签、计算器、
//    按名检索、实时编码面板）一个都写不出来，俄罗斯方块也只能用触屏虚拟按键
//    （app/tetris.scm 的已知取舍）。本库把键盘输入以【边沿事件 + 电平】两种形态
//    暴露给 Scheme，补齐最后一条输入通道。
//
//  与 GUI 输入管线的关系（重要）：
//    这是【旁路】，不是接管。HAL 在把按键注入 GUI 电平槽（input_device_read_key）
//    的同时，另投一份边沿事件进 nano_key_events 队列；GUI 的 get_input_event /
//    长按连发 / 4x4 宫格映射 / 软复位 全部照常工作，两边的行为都确定。
//    因此 App 运行期间 **ESC 依然是宿主的返回键**（触屏右上角宫格同理）。
//
//  事件模型（与 am_native_Touch 同一哲学，但数据源不同）：
//    Touch 的 HAL 只给电平，因此边沿由本库侧「采样差分」派生（亚帧点按可能漏检）；
//    Key 的 HAL 直接给边沿（nano_key_events），故按下/松开各**恰好一次**，
//    亚帧点按不漏。本库再把两种边沿分栈存放，于是 take_press / take_release
//    互不吞噬：App 可以先取若干个按下沿、再取松开沿。
//
//  接口（全部值型，按单值栈纪律 N1/N6：压 1 值 + step 1 次）：
//
//    (Key.is_available) : Boolean
//        本平台的键盘 HAL 是否接入了事件源（nano_key_events_mark_source）。
//        为 #f 时其余过程安全降级（take_* 恒 0、is_down 恒 #f），App 可据此
//        退回纯触屏交互。注意它表达的是「能力」，不是「此刻有键按下」。
//
//    (Key.is_down key_code) : Boolean
//        key_code 是 hal_key.h 的 NANO_KEY_* 逻辑键值：可打印字符就是其 ASCII
//        码（A=65、a=97、space=32、0..9=48..57、! =33…），另有
//        up=1 down=2 left=3 right=4 esc=27 enter=13 backspace=8 tab=9。
//        与「队列是否已被消费」无关，永远反映当拍真实电平（适合做按住加速、
//        组合键）。
//
//    (Key.take_press) : Number
//        取走一个最早的「按下」事件，返回其键码；没有待取事件时返回 0
//        （0 = NANO_KEY_IDLE，不可能是有效键码）。
//
//    (Key.take_release) : Number
//        同上，但取「松开」事件。松开事件对"按下沿武装、松手沿结算"的交互
//        （俄罗斯方块/虚拟摇杆式操作）不可缺。
//
//    (Key.dropped) : Number
//        因内部队列溢出而被丢弃的事件条数（诊断用，正常使用恒 0）。
//
//  Scheme 侧用法：源码中声明 (native Key) 后以 Key.函数名 调用。
//  文本输入的惯用写法（可打印字符即 ASCII）：
//      (define k (Key.take_press))
//      (if (and (> k 31) (< k 127)) (set! text (String.concat text (String.fromCharCode k))))
////////////////////////////////////////////////////////////////////////////

// 由本文件定义
extern const am_native_lib_entry_t am_native_Key_lib;

// (Key.is_available) : Boolean —— 本平台是否提供按键事件源
int32_t am_native_Key_is_available(am_runtime_t *rt, am_process_t *proc);

// (Key.is_down key_code:Number) : Boolean —— 该键当前是否按住
int32_t am_native_Key_is_down(am_runtime_t *rt, am_process_t *proc);

// (Key.take_press) : Number —— 取走一个按下沿（无则 0）
int32_t am_native_Key_take_press(am_runtime_t *rt, am_process_t *proc);

// (Key.take_release) : Number —— 取走一个松开沿（无则 0）
int32_t am_native_Key_take_release(am_runtime_t *rt, am_process_t *proc);

// (Key.dropped) : Number —— 溢出丢弃的事件累计条数
int32_t am_native_Key_dropped(am_runtime_t *rt, am_process_t *proc);

// （宿主用，非 Scheme 可见）最近一次被 Scheme 取走的按下沿的键码与时刻（ms）。
// 用途：模态文本输入框打开后，要丢弃"打开它的那次按键"迟到的松开沿（按键边沿由生产端
// 逐事件投递、消费者每帧取一个，按下与松开会跨帧），否则输入框会一出现就被提交。
// 无记录（从未取过按下沿）时 *code 置 0、*timestamp_ms 置 0。
void am_native_Key_last_press(uint8_t *code, uint64_t *timestamp_ms);

// （宿主用，非 Scheme 可见）清空事件源与本地事件栈。
// 每次启动一个灵机 App 时由宿主调用：App 未运行期间（菜单/其他界面）键盘事件
// 会积在旁路队列里，若不清空，新启动的 App 会先收到一串陈旧按键。
void am_native_Key_reset(void);

#ifdef __cplusplus
}
#endif

#endif