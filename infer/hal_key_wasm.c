// hal_key_wasm.c —— Nano-WASM-GUI 键盘 HAL（Emscripten）
//
// 无实体键盘平台（NANO_HAS_HW_KEYBOARD=0），主交互为触屏（页面 Pointer Events）。
// 本 HAL 额外提供「浏览器物理键盘」通路：页面 JS 的 keydown/keyup 事件经 ccall
// 注入（wasm_gui_on_key），供桌面浏览器用方向键/回车/Esc/数字十六键等操作。
//
// 事件模型（2026-10-14 重写，修复按键处理缺陷）：
// 参照 main.c 拆分事件环路「生产者高频采样/可靠投递，消费者按帧消费」的精神，
// 在单线程 WASM 下的等价实现——JS 事件回调即生产者（浏览器事件循环驱动，与 rAF
// 帧率解耦），get_input_event 的电平轮询即消费者：
//   - 电平：keydown 置位、keyup 清除（浏览器 keydown 重复事件重复置位无害），
//     上层既有的按下计时/长按(-2)/按住连发/软复位逻辑全部照常工作（此前一次性
//     槽位实现没有按住电平，长按永远不触发，且未监听 keyup）；
//   - 亚帧短按保底（s_key_unread）：按下后若 keyup 先于下一次轮询（快速点按，
//     电平从未被采到），补报一次按下使边沿检测看到完整的按下→松开（与触屏 HAL
//     的 down_pending 锁存同一哲学）；电平一旦被采样即清除未读标记，防止长按
//     松开后多报一次短按。

#include "hal_key.h"

#include "platform.h"

#include <emscripten.h>

static volatile uint8_t s_key_down   = NANO_KEY_IDLE; // 当前按住的键（电平）
static volatile uint8_t s_key_unread = NANO_KEY_IDLE; // 未被轮询消费过的按下（亚帧点按保底）

// 键盘事件入口（页面 JS ccall）：key_code 为 NANO_KEY_* 逻辑键值（hal_key.h）；
// is_down：1=按下（keydown） 0=松开（keyup）
EMSCRIPTEN_KEEPALIVE
void wasm_gui_on_key(int32_t key_code, int32_t is_down) {
    if (key_code <= NANO_KEY_IDLE || key_code > 127) return;
    if (is_down) {
        s_key_down = (uint8_t)key_code;
        s_key_unread = (uint8_t)key_code; // 新按下标记未读
    }
    else if (s_key_down == (uint8_t)key_code) {
        s_key_down = NANO_KEY_IDLE; // 松开（仅清除当前按住键；其他键的 keyup 忽略）
    }
}

int32_t input_device_init() {
    return 0;
}

uint8_t input_device_read_key() {
    if (s_key_down != NANO_KEY_IDLE) {
        s_key_unread = NANO_KEY_IDLE; // 电平被采样即视为本次按下已读
        return s_key_down;
    }
    if (s_key_unread != NANO_KEY_IDLE) {
        // 亚帧点按（DOWN/UP 落在同一帧间隔）：补报一次按下，下一帧恢复 IDLE
        uint8_t key = s_key_unread;
        s_key_unread = NANO_KEY_IDLE;
        return key;
    }
    return NANO_KEY_IDLE;
}
