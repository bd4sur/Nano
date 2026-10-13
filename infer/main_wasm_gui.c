// main_wasm_gui.c —— Nano-WASM-GUI 主入口（Emscripten）
//
// 与 main.c 串行单循环路径等价，差异仅在主循环驱动方式：
// 浏览器中不能无限 while(1)（卡死事件循环），改用 emscripten_set_main_loop
// 注册每帧回调（requestAnimationFrame 节奏，与页面渲染天然同步）。
// 帧回调体与 main.c 串行循环的单次迭代逐行一致。

#include <locale.h>
#include <stdlib.h>
#include <time.h>

#include <emscripten.h>

#include "platform.h"
#include "ui_app.h"
#include "graphics.h"

// 全局UI状态（与 main.c 同）
static Global_State *global_state = NULL;
static Key_Event    *key_event = NULL;

// 每帧回调：与 main.c 串行循环的单次迭代一致
static void wasm_gui_frame(void) {
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
