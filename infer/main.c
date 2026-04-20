#include <locale.h>
#include "platform.h"
#include "ui_app.h"

// 全局UI状态

static Global_State *global_state = NULL;
static Key_Event    *key_event = NULL;


int main() {
    if(!setlocale(LC_CTYPE, "")) return -1;

    key_event = (Key_Event*)platform_calloc(1, sizeof(Key_Event));
    global_state = (Global_State*)platform_calloc(1, sizeof(Global_State));

    main_init(key_event, global_state);

    while (1) {
        // 物理时间戳
        global_state->timestamp = get_timestamp_in_ms();
        // 获取按键事件
        get_key_event(key_event, global_state);
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
