// hal_key_wasm.c —— Nano-WASM-GUI 键盘 HAL（Emscripten）
//
// 无实体键盘平台（NANO_HAS_HW_KEYBOARD=0），主交互为触屏（页面 Pointer Events）。
// 本 HAL 额外提供「浏览器物理键盘」便利通路：页面 JS keydown 经 ccall 注入
// NANO_KEY_* 键码（单槽缓存，input_device_read_key 每次取一码），
// 供桌面浏览器用方向键/回车/Esc 等操作（行为等价于实体键盘按键）。

#include "hal_key.h"

#include "platform.h"

#include <emscripten.h>

// 单槽键码缓存（JS 注入 → input_device_read_key 取走；NANO_KEY_IDLE=空）
static volatile uint8_t s_key_slot = NANO_KEY_IDLE;

// 键盘事件入口（页面 JS ccall）：key_code 为 NANO_KEY_* 逻辑键值（hal_key.h）
EMSCRIPTEN_KEEPALIVE
void wasm_gui_on_key(int32_t key_code) {
    if (key_code <= NANO_KEY_IDLE || key_code > 127) return;
    s_key_slot = (uint8_t)key_code; // 已占用时被覆盖（高频连发场景与实体键盘扫描同语义）
}

int32_t input_device_init() {
    return 0;
}

uint8_t input_device_read_key() {
    uint8_t key = s_key_slot;
    s_key_slot = NANO_KEY_IDLE;
    return key;
}
