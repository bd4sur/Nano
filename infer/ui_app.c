#include <stdio.h>
#include <time.h>

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    #include "esp_heap_caps.h"
    // 内置模型权重头（nano_psycho_230k_q80.h）已随模型加载逻辑移至 ui_llm.c
#endif

#include "hal_display.h" // display_sleep/wakeup/set_brightness：锁屏路径（NANO_HAS_PWR_KEY）使用；
                         // 原仅 ESP32 分支包含，其余平台靠隐式声明（clang 下 -Werror 报错），统一包含

#include "graphics.h"
#include "hal_key.h"
#include "hal_touch.h" // hal_touch_wheel_take：滚轮增量取走（仅 NANO_WASM_GUI 编译分支调用）
#include "ui.h"
#include "ui_softkbd.h"
#include "ui_grid16kbd.h"
#include "ui_pinyin_ime.h"
#include "ui_settings.h"

#include "platform.h"

#include "hal_audio_out.h" // audio_out_set_master_volume：系统设置改音量时应用硬件
#include "hal_misc.h"       // misc_led_blink/misc_led_set：锁屏 LED 心跳

#include "infer.h"

#ifdef IMU_ENABLED
    #include "hal_imu.h"
#endif

#ifdef UPS_ENABLED
    #include "hal_power.h"
#endif

#ifdef ASR_ENABLED
    #include "asr.h"
#endif

#ifdef TTS_ENABLED
    #include "tts.h"
#endif


#include "ui_flip.h"    // 番茄表/电子沙漏（原在本文件内的 STATE_FLIP 界面与交互）

#include "ui_genetic.h"

#include "ui_cloud.h"
#include "ui_llm.h"
#include "ui_linglong.h"   // 玲珑天象仪（原在本文件内的 STATE_LINGLONG 状态页）
#include "ui_main_menu.h"  // 主菜单（原在本文件内的 STATE_MAIN_MENU 表驱动宫格）
#include "nongli.h"

#include "ui_color.h"
#include "ui_app.h"

#include "ui_animac.h"

#include "ui_ripple.h"

#include "ui_scheme_app.h"

#include "ui_water.h"

#include "ui_ebook.h"

#include "ui_ofdm.h"

#include "ui_musicbox.h"

#include "ui_dict.h"

#define WALLPAPER_PATH (PLATFORM_ROOT_DIR "/wallpaper/wp.png")

// 全局变量（TODO 临时，后续要全部移到全局状态上下文中）

static uint64_t last_splash_timestamp = 0;

// 指向图像缓冲区的指针
static uint8_t *s_image_file_buffer = NULL;
static size_t s_image_file_size = 0;
static char s_image_filename_buffer[128]; // 缓存图像文件名，用于确定是否要重新读取、重新解码
// 壁纸图像解码后的 RGB888 像素缓冲区（避免每次渲染都重新解码）
static uint8_t *s_image_rgb888_buffer = NULL;
static uint32_t s_image_width = 0;
static uint32_t s_image_height = 0;
static uint8_t s_image_decode_ready = 0;



static uint32_t s_animac_prev_ui_font = 0; // 进入 STATE_ANIMAC_* 之前的 ui_font，退出时恢复
// 控制台触屏序列属主：0=无序列 1=日志区 2=输入框 3=穿透（输入框无滚动余量：拖滚日志区+点按定位光标）
//（DOWN 时按按下点归属，整个序列只喂给属主手势机）
static int32_t s_animac_touch_owner = 0;

// 控制台退出确认对话框的实例（全局确认模态框组件，见 ui.h）。
// 弹出时由 ui_confirm_modal_init 重置文案与按下态，存续期间每帧喂触点。
static Ui_Confirm_Modal s_animac_exit_modal;






// ===============================================================================
// UI框架：获取输入事件（按键 + 触屏）
//
// 事件层只提供原始输入：实体按键、显式虚拟键盘（触屏软键盘 ui_softkbd / 16 键虚拟键盘
// ui_grid16kbd）派生的键码、触屏电平/坐标（key_event->touch_*）与 DOWN/UP 边沿
// （key_event->touch_edge）。触屏手势的识别与语义解释归消费者（通用跟踪器
// ui_swipe_tracker_*，见 ui.c），本层不参与。
//
// 注：触屏不再派生任何“硬按键”事件（历史上的全屏 4x4 宫格隐式软键映射已整体退场，
// 2026-11）。需要键码的触屏输入只有一个来源——显式绘制的虚拟键盘（软键盘/16 键键盘），
// 其键码经 Key_Event.is_soft_key=1 打标，与实体键盘（is_soft_key=0）区分。
// ===============================================================================

// ===============================================================================
// 输入事件处理：背景、现状与 AI 必读原则
//
// 【历史背景】
// 项目最初只有实体十六键，全部业务基于 NANO_KEY_* 键码事件。支持触屏后为平滑过渡，
// 曾由本层把触屏位置按屏幕 4x4 宫格映射为键码（“软按键”），使旧键码逻辑在纯触屏
// 设备上照常工作；该机制已于 2026-11 整体退场（各状态改为直接消费触屏流）。
//
// 【现状】
// 触屏只有一条到达通道：触点电平/坐标/边沿（key_event->touch_* / touch_edge），
// 经跨核共享快照 + 可靠边沿事件投递。键码事件则有两个来源：实体键盘
// （is_soft_key=0）与显式虚拟键盘（is_soft_key=1，软键盘另置 is_softkbd=1）。
//
// 【按键/触屏事件处理原则】（所有功能开发与修改必须遵守）
//   1. 存量功能默认仍按硬按键语义工作。仅应响应硬按键的功能，必须显式过滤
//      key_event->is_soft_key == 0（范式：ui_flip_event_handler），
//      否则触屏虚拟键盘的键码会被误当作实体键响应。
//   2. 需要直接消费触屏的功能状态（虚拟按钮/点按/拖动等）：点按边沿认
//      key_event->touch_edge（TOUCH_EDGE_DOWN/UP，生产端高频检测+队列可靠投递，
//      亚帧点按不湮灭），按下点坐标取 touch_down_x/y（范式：ui_flip.c、
//      ui.c 菜单控件）；拖动轨迹逐帧读 key_event->touch_x/touch_y
//      快照。一律不得跨层直读 hal_touch，也不再需要模块自持 prev 电平做沿检测。
//   3. 触屏是唯一入口的状态（无实体键盘的机型上），必须自备显式触屏入口——
//      至少一个“返回/退出”热点，否则该状态在纯触屏设备上是死路。
//   4. 会引发状态切换的触屏动作，必须在松手沿（touch_edge & TOUCH_EDGE_UP）
//      触发；按下沿只允许锁存坐标/状态。如此状态切换发生在手指抬起之后，
//      下一状态看到的触屏电平为 0，不会将本次触摸序列误当作新的点击
//      （范式：ui.c 菜单控件、ui_flip.c）。
// ===============================================================================
void get_input_event(Key_Event *key_event, Global_State *global_state) {
    // 触屏边沿检测状态（本函数在 Core1 以 1-2ms 轮询，静态变量天然单生产者）
    static int32_t s_input_touch_prev = 0;   // 上一轮询的触屏电平（经 UP 去抖后的认定值）
    static int32_t s_input_touch_down_x = 0; // 本次触摸序列按下点坐标
    static int32_t s_input_touch_down_y = 0;
    static int32_t s_input_touch_last_x = 0; // 触摸期间最后有效触点（UP 事件的松开坐标）
    static int32_t s_input_touch_last_y = 0;
    static int32_t s_input_touch_up_pending = 0; // UP 去抖：已连续无按压的轮询数
    // 实体按键读取（无实体键盘的触屏设备恒为 NANO_KEY_IDLE）：
    // 部分平台需在本调用内完成输入流解复用（ncurses：drain 鼠标事件并转发触屏HAL缓存），
    // 故须在触屏采样之前调用，保证下方的触屏样本为本帧最新
    uint8_t key = input_device_read_key();
    uint8_t key_is_soft = 0; // 按键来源：0-实体键盘，1-显式虚拟键盘（软键盘/16键键盘）

    // 触屏统一采样（本函数在 Core1 每 1-2ms 轮询一次）：坐标与电平填入 key_event，
    // 供本轮所有消费者（虚拟键盘、手势解释及各业务状态）使用，上层不再直接调 touch_read
    touch_read(&key_event->touch_x, &key_event->touch_y, &key_event->is_touching);

    // 触屏 UP 沿去抖（须在共享快照与边沿检测之前，保证三者语义一致）：松开认定要求连续
    // NANO_TOUCH_UP_DEBOUNCE 次轮询均无按压。背景：触屏挂在多设备共享的 I2C 总线上（Core2 的
    // FT6336 与 PMIC/IMU 同总线），拖动中偶发单次读抖动会产生假松开。若不去抖，假 UP+DOWN
    // 对会把一次拖动切成“点按（光标跳变）+重激活（拖动位移阈值重新累计）”循环，
    // 表现为触屏拖动失灵而点按定位正常（2026-08 定位）。抖动窗口内按按住对待
    //（key_event 为静态变量，未按压时 touch_read 不写坐标，自然保持上轮最后有效值）。
#if defined(NANO_WASM_GUI)
    // WASM 特例：触屏电平来自页面 Pointer Events 的 pointerdown/up，浏览器保证一次
    // 指针序列的按下/松开边沿唯一，不存在 Core2 共享 I2C 总线上的单次读抖动，因此
    // 不需要 N 次轮询去抖。若沿用去抖，WASM 的生产者只在事件到达 + 50ms 心跳时采样，
    // pointerup 后要等两次心跳（50~100ms）才产生 UP 沿；这期间 rAF 消费者仍按“按住”
    // 逐帧跑拖动速度 EMA（零位移 ×0.6/帧），把甩动初速度衰减到约 5%，菜单/文本控件
    // 的松手惯性因此消失（2026-11 定位）。松开沿立即生效（保留静态变量以便结构一致）。
    s_input_touch_up_pending = 0;
#else
    if (!key_event->is_touching && s_input_touch_prev) {
        if (s_input_touch_up_pending + 1 < NANO_TOUCH_UP_DEBOUNCE) {
            s_input_touch_up_pending++;
            key_event->is_touching = 1;
        }
    }
    else {
        s_input_touch_up_pending = 0;
    }
#endif

    // 触屏电平共享快照：ESP32 上 Core0 渲染任务每帧取用本快照覆盖到其 key_event，
    // 高频电平样本不进入事件队列（见 linglong_m5core2.ino）
    global_state->touch_x = key_event->touch_x;
    global_state->touch_y = key_event->touch_y;
    global_state->is_touching = key_event->is_touching;
#if defined(NANO_WASM_GUI)
    // 滚轮/触控板增量：与触屏电平同理走“高频生产、按帧消费”——本处只累加，
    // 由帧回调（main_wasm_gui.c）取走并清零后填入 key_event->wheel_dy。
    // 不进事件队列：滚轮是逐帧消费的连续量，队列留给一次性边沿。
    global_state->wheel_dy += hal_touch_wheel_take();
#endif

    // 触屏边沿检测（生产端，与按键边沿同一哲学：消费者零负担，见 AGENTS.md 第八节）：
    // DOWN/UP 边沿填入 touch_edge，经事件队列可靠投递（见 .ino loop 与 core0_render_task）；
    // 移动轨迹仍走上方共享快照，不入队。DOWN 时锁存按下点坐标；触摸期间持续记录最新
    // 触点，供 UP 事件携带有效的松开坐标（松开瞬间 touch_read 坐标不保证有效）。
    key_event->touch_edge = 0;
    if (key_event->is_touching) {
        if (!s_input_touch_prev) {
            key_event->touch_edge = TOUCH_EDGE_DOWN;
            s_input_touch_down_x = key_event->touch_x;
            s_input_touch_down_y = key_event->touch_y;
        }
        s_input_touch_last_x = key_event->touch_x;
        s_input_touch_last_y = key_event->touch_y;
    }
    else if (s_input_touch_prev) {
        key_event->touch_edge = TOUCH_EDGE_UP;
        key_event->touch_x = s_input_touch_last_x; // 松开坐标不保证有效，以最后有效触点代替
        key_event->touch_y = s_input_touch_last_y;
        s_input_touch_up_pending = 0; // UP 沿确认，去抖计数复位
    }
    s_input_touch_prev = key_event->is_touching;
    key_event->touch_down_x = s_input_touch_down_x;
    key_event->touch_down_y = s_input_touch_down_y;

    // 触屏本身不再派生键码（全屏 4x4 宫格隐式软键映射已整体退场，2026-11）。
    // 触屏派生的键码只有一个来源：显式绘制在屏幕上的虚拟键盘（见下）。
    uint8_t key_is_softkbd = 0;

    // 16键虚拟键盘（文本输入控件固有输入法）：可见时，键盘区域内的触摸由键盘接管——
    // 命中按钮产生对应键码（走下方统一的边沿/长按机制）；键盘区域外的触屏不产生键码。
    if (ui_grid16kbd_is_visible()) {
        uint8_t grid16_key = ui_grid16kbd_poll(key_event->touch_x, key_event->touch_y, key_event->is_touching);
        if (ui_grid16kbd_touch_claimed()) {
            key = grid16_key;
            key_is_soft = (grid16_key != NANO_KEY_IDLE); // 16键虚拟键盘键码来自触屏
        }
        else {
            key = NANO_KEY_IDLE; // 键盘区域外：不产生键码
        }
    }

    // 触屏软键盘：可见时，键盘区域内的触摸由软键盘接管——注入软键盘键码
    //（无边沿时为NANO_KEY_IDLE，仍走下方的边沿检测机制）；键盘区域外的触屏不产生键码
    //（实体键盘的按键不受影响）。
    if (ui_softkbd_is_visible()) {
        uint8_t softkbd_key = ui_softkbd_poll(key_event->touch_x, key_event->touch_y, key_event->is_touching);
        if (ui_softkbd_touch_claimed()) {
            key = softkbd_key;
            key_is_soft = (softkbd_key != NANO_KEY_IDLE); // 软键盘键码来自触屏
            key_is_softkbd = (softkbd_key != NANO_KEY_IDLE);
        }
    }

    // 边沿
    if (key_event->key_mask != 1 && (key != key_event->prev_key)) {
        // 按下瞬间（上升沿）
        if (key != NANO_KEY_IDLE) {
            key_event->key_code = key;
            key_event->key_edge = 1;
        }
        // 松开瞬间（下降沿）
        else {
            key_event->key_code = key_event->prev_key;
            // 短按（或者通过长按触发重复动作状态后反复触发）
            if (key_event->key_repeat == 1 ||
                ((global_state->timestamp - key_event->key_timer) >= 0 &&
                    (global_state->timestamp - key_event->key_timer) < LONG_PRESS_THRESHOLD)) {
                key_event->key_edge = -1;
            }
            // 长按
            else if ((global_state->timestamp - key_event->key_timer) >= LONG_PRESS_THRESHOLD) {
                key_event->key_edge = -2;
                key_event->key_repeat = 1;
            }
        }
        key_event->key_timer = global_state->timestamp;
    }
    // 按住或松开
    else {
        // 按住
        if (key != NANO_KEY_IDLE) {
            key_event->key_code = key;
            key_event->key_edge = 0;
            // key_event->key_timer++;
            // 若重复动作标记key_repeat在一次长按后点亮，则继续按住可以反复触发短按
            if (key_event->key_repeat == 1) {
                key_event->key_edge = -2;
                key_event->key_mask = 1; // 软复位置1，即强制恢复为无按键状态，以便下一次轮询检测到下降沿（尽管物理上有键按下），触发长按事件
                key = NANO_KEY_IDLE; // 便于后面设置prev_key为KEY_IDLE（无键按下）
                key_event->key_repeat = 1;
            }
            // 如果没有点亮动作标记key_repeat，则达到长按阈值后触发长按事件
            else if ((global_state->timestamp - key_event->key_timer) >= LONG_PRESS_THRESHOLD) {
                key_event->key_edge = -2;
                key_event->key_mask = 1; // 软复位置1，即强制恢复为无按键状态，以便下一次轮询检测到下降沿（尽管物理上有键按下），触发长按事件
                key = NANO_KEY_IDLE; // 便于后面设置prev_key为KEY_IDLE（无键按下）
            }
        }
        // 松开
        else {
            key_event->key_code = NANO_KEY_IDLE;
            key_event->key_edge = 0;
            key_event->key_timer = global_state->timestamp;
            key_event->key_mask = 0;
            key_event->key_repeat = 0;
        }
    }
    if (key != NANO_KEY_IDLE) {
        key_event->is_soft_key = key_is_soft;   // 记录按键来源（软-触屏派生/硬-实体键盘），下降沿事件沿用它
        key_event->is_softkbd = key_is_softkbd; // 记录按键来源，下降沿事件沿用它
    }
    key_event->prev_key = key;
}


// ===============================================================================
// UI框架：全局GUI+gfx初始化
// ===============================================================================

// 注：软键盘显隐切换已下沉为文本输入控件固有功能（ui.c 的 ui_widget_input_toggle_softkbd，
// 供 Ctrl+0 组合键与上滑/下滑手势在任何输入状态使用）。

void ui_init(Key_Event *key_event, Global_State *global_state) {

    global_state->w_textarea_main = (Widget_Textarea_State*)platform_calloc(1, sizeof(Widget_Textarea_State));

    global_state->w_input_main = (Widget_Input_State*)platform_calloc(1, sizeof(Widget_Input_State));

    global_state->w_menu_main = (Widget_Menu_State*)platform_calloc(1, sizeof(Widget_Menu_State));


    global_state->STATE = STATE_SPLASH_SCREEN;
    global_state->PREV_STATE = STATE_DEFAULT;

    global_state->ui_color_style = UI_COLOR_DARK;

    // 全局默认背光（与 display_hal_init 的 setBrightness 一致；设置菜单按此值显示/调节）
    global_state->brightness = NANO_DEFAULT_BRIGHTNESS;

    // 全局主音量（影响按键音、寻呼机发射音量、音乐盒初始音量）
    global_state->volume = 16;
    // 自动关机（默认关）
    global_state->auto_shutdown_minutes = 0;
    global_state->auto_shutdown_deadline = 0;
    global_state->key_feedback_mode = 1;  // 按键提示（按键反馈方式）默认为灯光
    global_state->splash_clock_style = SPLASH_CLOCK_SMALL;  // 启动画面时钟样式默认为“小”（时间并入顶部日期行）

    global_state->timestamp_last = 0;
    global_state->touch_x = 0;
    global_state->touch_y = 0;
    global_state->is_touching = 0;

    global_state->is_ctrl_enabled = 0;
    // 鹦鹉笼（LLM）相关字段初始化已提取至 ui_llm 模块
    ui_llm_init_config(global_state);
#ifdef ASR_ENABLED
    global_state->asr_output_buffer = (wchar_t*)platform_calloc(UI_STR_BUF_MAX_LENGTH, sizeof(wchar_t));
    wcscpy(global_state->asr_output_buffer, L"请说话...");
#else
    global_state->asr_output_buffer = NULL;
#endif
    global_state->is_auto_submit_after_asr = 1; // ASR结束后立刻提交识别内容到LLM
    global_state->is_asr_server_up = 0;
    global_state->is_recording = 0;
    global_state->asr_start_timestamp = 0;
    global_state->pitch = 0.0f;
    global_state->roll = 0.0f;
    global_state->yaw = 0.0f;
    global_state->imu_temperature = 0.0f;
    global_state->is_full_refresh = 1;

    global_state->ui_font = GFX_FONT_ALPHA_16;
}


void ui_draw_image(Key_Event *key_event, Global_State *global_state, const char *img_path, int32_t is_reload) {

    int32_t is_new_image_file = (strcmp(img_path, s_image_filename_buffer) != 0);

    // 如果显式指定reload，或者图像文件名跟上次调用不同，则清除缓冲区，重新读文件并解码
    if (is_reload || is_new_image_file) {
        printf("reload/refresh image: %s\n", img_path);
        strcpy(s_image_filename_buffer, img_path);
        s_image_decode_ready = 0;
        if (s_image_file_buffer != NULL) {
            free(s_image_file_buffer);
            s_image_file_buffer = NULL;
        }
        if (s_image_rgb888_buffer != NULL) {
            free(s_image_rgb888_buffer);
            s_image_rgb888_buffer = NULL;
        }
    }

    // 首次加载：从SD卡读取图像文件到文件缓冲区
    if (s_image_file_buffer == NULL) {
        int32_t ret = platform_read_file_to_buffer(img_path, &s_image_file_buffer, &s_image_file_size);
        // printf("platform_read_file_to_buffer %d\n", ret);
    }

    // 首次解码：将图像文件解码到 RGB888 像素缓冲区（避免每次渲染都重新解码）
    if (s_image_file_buffer != NULL && s_image_file_size > 0 && !s_image_decode_ready) {
        if (s_image_rgb888_buffer == NULL) {
            s_image_rgb888_buffer = (uint8_t *)platform_malloc(SCREEN_WIDTH * SCREEN_HEIGHT * 3);
        }
        if (s_image_rgb888_buffer != NULL) {
            int32_t ret = gfx_decode_image_buffer(
                s_image_file_buffer, s_image_file_size,
                SCREEN_WIDTH, SCREEN_HEIGHT,
                s_image_rgb888_buffer,
                &s_image_width, &s_image_height
            );
            if (ret == 0) {
                s_image_decode_ready = 1;
                printf("gfx_decode_image_buffer ok, %dx%d\n", s_image_width, s_image_height);
            } else {
                printf("gfx_decode_image_buffer failed\n");
            }
        }
    }

    // 优先使用已解码的 RGB888 缓冲区绘制壁纸
    if (s_image_decode_ready && s_image_rgb888_buffer != NULL) {
        gfx_draw_rgb888_buffer(
            global_state->gfx, s_image_rgb888_buffer,
            s_image_width, s_image_height,
            0, 0
        );
    }
    // 若解码尚未完成或失败，回退到原始方式（带实时解码）
    else if (s_image_file_buffer != NULL && s_image_file_size > 0) {
        gfx_draw_image_buffer(global_state->gfx, s_image_file_buffer, s_image_file_size, 0, 0, 0, 0);
        printf("gfx_draw_image_buffer\n");
    }
}




void init_game_menu(Key_Event *key_event, Global_State *global_state) {
    // 条目字符串借用字面量的静态存储，控件不复制
    static const wchar_t *game_menu_items[] = {
        L"灵机应用程序",
        L"遗传算法拟合图像",
        L"水波",
        L"水池",
        // 原“黄金矿工”入口已移除：C 版（STATE_GOLDMINER / ui_goldminer.c）退场，
        // 黄金矿工仅保留 Scheme 实现（app/goldminer.scm，经“灵机应用程序”入口启动）
        // 原“遗传算法求解旅行商问题”入口已移除：C 版（STATE_GENETIC_TSP / ui_tsp.h）
        // 退场，TSP 仅保留 Scheme 实现（app/tsp.scm，经“灵机应用程序”入口启动）
        // 原“日历”入口已移除：C 版（STATE_CALENDAR / ui_calendar.c）退场，日历仅保留
        // Scheme 实现（app/calendar.scm，经“灵机应用程序”入口启动）
        // L"体积云",  // 原入口：已移除（体积云渲染保留在天象仪“大气散射”档位中，见 celestial.c）
        // L"曼德博集" / L"生命游戏Scheme"：已并入“灵机应用程序”（业务资源目录 app/
        // 下的 Scheme App 动态列表，见 ui_scheme_app.c）
        // L"小鹦鹉笼",  // 原入口：小鹦鹉笼已并入“鹦鹉笼”的统一模型菜单（[轻] 前缀条目），注释保留备查
        // 原“俄罗斯方块”入口已移除：C 版（STATE_TETRIS）退场，俄罗斯方块仅保留
        // Scheme 实现（app/tetris.scm，经“灵机应用程序”入口启动）
        // 原“康威生命游戏”入口已移除：C 版（STATE_GAMEOFLIFE）退场，生命游戏仅保留
        // Scheme 实现（app/lib/alife_gol.scm —— 2026-11 起是 app/emergence.scm 的
        // 「生命游戏」页，不再是独立 App）
        // 原“粒子生命”入口已移除：C 版（STATE_PARTICLELIFE）退场，粒子生命仅保留
        // Scheme 实现（app/lib/alife_particle.scm —— 同上，已并入 app/emergence.scm）
        // 原“Bad Apple!”入口已移除：C 版（STATE_BADAPPLE）退场，Bad Apple 仅保留
        // Scheme 实现（app/badapple.scm，经“灵机应用程序”入口启动）
    };
    global_state->w_menu_main->title = L"小游戏";
    global_state->w_menu_main->items = game_menu_items;
    global_state->w_menu_main->item_num = 4;
    ui_widget_menu_init(key_event, global_state, global_state->w_menu_main);
}

int32_t game_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms) {
    switch (ms->current_item_index) {
        case 0: return STATE_SCHEME_APP_LIST;  // 灵机应用程序（Scheme App 动态列表）
        case 1: return STATE_GENETIC;
        case 2: return STATE_RIPPLE;
        case 3: return STATE_WATER;
        // 原 case 1（STATE_GOLDMINER）已移除：C 版黄金矿工退场，仅保留 app/goldminer.scm
        // 原 case 2（STATE_GENETIC_TSP）已移除：C 版 TSP 退场，仅保留 app/tsp.scm
        // 原 case 6: 原“日历”入口已移除（C 版退场，日历仅保留 Scheme 实现 app/calendar.scm）
        // case 7: 原“体积云”入口已移除（保留 STATE_CLOUD 状态处理，当前无菜单入口）
        // 原“曼德博集”/“生命游戏Scheme”入口已并入“灵机应用程序”（case 0）
        // case 12: 原“小鹦鹉笼”入口已移除（并入鹦鹉笼模型菜单，条目带 [轻] 前缀）
        default: return STATE_MAIN_MENU;
    }
}













// ===============================================================================
// 开机欢迎画面
// ===============================================================================

// 绘制点阵的版权信息
//   点阵数据通过bd4sur.com/html/am32.html取得
static void ui_draw_copyright_notice(Key_Event *key_event, Global_State *global_state, uint32_t x_offset, uint32_t y_offset) {
    uint32_t callsign[5]  = {3876120944, 2493057352, 3836266864, 2495359312, 3876177480}; // BD4SUR, width=29
    uint32_t year_2025[5] = {1662631936, 2493841408, 613010944, 1150296064, 4080910336};  // 2025-, width=23
    uint32_t year_2026[5] = {1662566400, 2493841408, 613007360, 1150361600, 4080844800};  // 2026, width=19
    uint32_t copy_mark[7] = {2013265920, 2214592512, 3019898880, 2751463424, 3019898880, 2214592512, 2013265920}; // (c), width=6

    uint32_t x = x_offset;

    for (uint32_t i = 0; i < 7; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            gfx_draw_point(global_state->gfx, (x + j), (y_offset + i), 255, 255, 255, (copy_mark[i] >> (32 - 1 - j)) & 0x1);
        }
    }
    x += (6 + 4);
    for (uint32_t i = 0; i < 5; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            gfx_draw_point(global_state->gfx, (x + j), (y_offset + 1 + i), 255, 255, 255, (year_2025[i] >> (32 - 1 - j)) & 0x1);
        }
    }
    x += (23 + 1);
    for (uint32_t i = 0; i < 5; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            gfx_draw_point(global_state->gfx, (x + j), (y_offset + 1 + i), 255, 255, 255, (year_2026[i] >> (32 - 1 - j)) & 0x1);
        }
    }
    x += (19 + 4);
    for (uint32_t i = 0; i < 5; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            gfx_draw_point(global_state->gfx, (x + j), (y_offset + 1 + i), 255, 255, 255, (callsign[i] >> (32 - 1 - j)) & 0x1);
        }
    }
}


// 启动画面底部三行（自下而上：日期时间 / 农历 / 电量信息）的排版常量
#define SPLASH_FOOTER_LINE_GAP      (6)  // 相邻两行之间的行间距（px）
#define SPLASH_FOOTER_BOTTOM_MARGIN (11) // 最底一行行框下边界距屏幕底边的留白（px，沿用旧电量行的留白）
// 是否在启动画面显示农历文本（1 = 显示，0 = 隐藏）。隐藏时农历不占行，
// 电量信息行紧贴在日期时间行上方。
#define SPLASH_SHOW_NONGLI          (1)

void ui_app_splash_render_frame(Key_Event *key_event, Global_State *global_state) {

    // 清屏
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_fill_white(global_state->gfx);
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_soft_clear(global_state->gfx);
    }

    // 绘制壁纸
    ui_draw_image(key_event, global_state, WALLPAPER_PATH, 0);

    // Header
    // ui_draw_header(key_event, global_state, L"Project Nano", 1);

    // 时间
    time_t rawtime;
    struct tm *timeinfo;
    time(&rawtime); // 获取当前时间戳
    timeinfo = localtime(&rawtime); // 转换为本地时间

    int32_t year = timeinfo->tm_year + 1900;
    int32_t month = timeinfo->tm_mon + 1;
    int32_t day = timeinfo->tm_mday;
    int32_t hour = timeinfo->tm_hour;
    int32_t minute = timeinfo->tm_min;
    int32_t second = timeinfo->tm_sec;
    double timezone = 8.0;
    double longitude = 119.0;
    double latitude = 32.0;

    wchar_t datetime_wcs_buffer[33];
    wchar_t nongli_wcs_buffer[33];

    uint8_t time_red = 0, time_green = 0, time_blue = 0;
    uint8_t nongli_red = 0, nongli_green = 0, nongli_blue = 0;
    uint8_t sevenseg_red = 0, sevenseg_green = 0, sevenseg_blue = 0;
    uint32_t sevenseg_shadow = 1;

    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        time_red = 255; time_green = 255; time_blue = 255;
        nongli_red = 0xff; nongli_green = 0xfb; nongli_blue = 0;
        sevenseg_red = 255; sevenseg_green = 255; sevenseg_blue = 255;
        sevenseg_shadow = 1;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        time_red = 255; time_green = 255; time_blue = 255;
        nongli_red = 0xff; nongli_green = 0xfb; nongli_blue = 0;
        sevenseg_red = 255; sevenseg_green = 255; sevenseg_blue = 255;
        sevenseg_shadow = 1;
    }

    // 启动画面文本统一使用 12px alpha 平滑字体（GFX_FONT_ALPHA_12），与全局 ui_font（16px）
    // 解耦，避免影响其他界面。
    // 底部三行排版（按屏幕高度自适应）：最底一行是日期时间，其上一行是农历，
    // 再上一行（仅 UPS_ENABLED）是电量信息；最底一行行框下边界距屏幕底边
    // SPLASH_FOOTER_BOTTOM_MARGIN，相邻两行之间留 SPLASH_FOOTER_LINE_GAP 行间距。
    // 农历隐藏（SPLASH_SHOW_NONGLI=0）时不占行，电量行直接接在日期时间行上方。
    // 居中绘制时行框下边界 = 行框中心 + (行高 - 行高/2)（见 gfx_font_draw_text_centered）。
    const int32_t nongli_line_h = gfx_font_line_height(GFX_FONT_ALPHA_12);
    const int32_t datetime_line_h = gfx_font_line_height(GFX_FONT_ALPHA_12);
    // 最底一行：日期时间（原农历所在位置）
    const int32_t datetime_center_y = global_state->gfx->height - SPLASH_FOOTER_BOTTOM_MARGIN
                                    - (datetime_line_h - datetime_line_h / 2);
    // 农历行：日期时间行的上一行，行框上边界与日期时间行下边界隔 SPLASH_FOOTER_LINE_GAP；
    // 农历隐藏时不占行（电量行的“农历行上一行”基准退化为日期时间行本身）。
    const int32_t nongli_center_y = SPLASH_SHOW_NONGLI
        ? (datetime_center_y - datetime_line_h / 2 - SPLASH_FOOTER_LINE_GAP
           - (nongli_line_h - nongli_line_h / 2))
        : datetime_center_y;

    const wchar_t *weekdays[] = {L"日", L"一", L"二", L"三", L"四", L"五", L"六"};
    // 日期时间行固定使用 12px alpha 平滑字体（窄屏也可完整显示）
    const uint32_t datetime_font = GFX_FONT_ALPHA_12;
    if (global_state->splash_clock_style == SPLASH_CLOCK_SMALL) {
        // 小样式：日期与 hh:mm:ss 合成一个字符串，整体居中
        swprintf(datetime_wcs_buffer, 33, L"%04d年%02d月%02d日 星期%ls %02d:%02d:%02d",
                 year, month, day, weekdays[timeinfo->tm_wday], hour, minute, second);
    }
    else {
        // 大样式：底部只显示日期，时间由画面中央的七段码时钟表现
        swprintf(datetime_wcs_buffer, 33, L"%04d年%02d月%02d日 星期%ls", year, month, day, weekdays[timeinfo->tm_wday]);
    }
    // 背景（阴影）
    gfx_font_draw_text_centered(global_state->gfx, datetime_font, datetime_wcs_buffer, global_state->gfx->width / 2 + 1, datetime_center_y + 1, 0x66, 0x66, 0x66, 1);
    // 前景
    gfx_font_draw_text_centered(global_state->gfx, datetime_font, datetime_wcs_buffer, global_state->gfx->width / 2, datetime_center_y, time_red, time_green, time_blue, 1);

    // 农历日期（日期时间行的上一行，12px alpha 平滑字体；可用 SPLASH_SHOW_NONGLI 隐藏）
    if (SPLASH_SHOW_NONGLI) {
        LunarDate *nongli = lunar_calculate(year, month, day, hour, minute, second, timezone);
        _mbstowcs(nongli_wcs_buffer, nongli->full_display, 33);
        // 背景（阴影）
        gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, nongli_wcs_buffer, global_state->gfx->width / 2 + 1, nongli_center_y + 1, 0x66, 0x66, 0x66, 1);
        // 前景
        gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, nongli_wcs_buffer, global_state->gfx->width / 2, nongli_center_y, nongli_red, nongli_green, nongli_blue, 1);
    }

    // 七段码时钟（仅“大”样式：绘制在画面中央）
    if (global_state->splash_clock_style == SPLASH_CLOCK_BIG) {
        wchar_t time7seg_str[10];
        swprintf(time7seg_str, 10, L"%02d:%02d:%02d", hour, minute, second);
        int32_t s7seg_width = 0.0f;
        int32_t s7seg_height = 0.0f;
        ui_draw_7seg_string_centered(key_event, global_state,
            global_state->gfx->width / 2 + 6, 55,
            time7seg_str, sevenseg_red, sevenseg_green, sevenseg_blue, 12.0f, 4.0f, 8.0f, sevenseg_shadow,
            &s7seg_width, &s7seg_height);
    }


    // 玲珑仪（青春版）
    // ui_linglong_draw_lite(key_event, global_state, (global_state->gfx->width - 128) / 2, 100,
    //     year, month, day, hour, minute, second, longitude, latitude, timezone);


    // Footer
    // if (global_state->gfx->width > 128) {
    //     ui_draw_footer(key_event, global_state, L"(c) 2025-2026 BD4SUR", 1);
    // }
    // else {
    //     ui_draw_copyright_notice(key_event, global_state, 20, 53);
    // }



    // 时间戳
    // wchar_t ts_text[100];
    // swprintf(ts_text, 100, L"Timestamp: %llu | Ticks: %d", global_state->timestamp, global_state->timer);
    // gfx_draw_textline_centered(global_state->gfx, ts_text, global_state->gfx->width/2, global_state->gfx->height-13*3-6, time_red, time_green, time_blue, 1);



#ifdef UPS_ENABLED
    // 电量信息行（农历行的上一行）：行框下边界 = 农历行行框上边界 - SPLASH_FOOTER_LINE_GAP
    const int32_t battery_line_h = gfx_font_line_height(GFX_FONT_ALPHA_12);
    int32_t battery_center_y = nongli_center_y - nongli_line_h / 2 - SPLASH_FOOTER_LINE_GAP
                             - (battery_line_h - battery_line_h / 2);
    if (battery_center_y < battery_line_h / 2) {
        battery_center_y = battery_line_h / 2; // 极小屏兜底：不越出屏幕上边界
    }

    // 绘制电池电量
    uint32_t icon_x = global_state->gfx->width - 17;
    uint32_t icon_y = 3;
    gfx_draw_line(global_state->gfx, (icon_x+1),  (icon_y),   (icon_x+14), (icon_y),   255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x+14), (icon_y),   (icon_x+14), (icon_y+7), 255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x+1),  (icon_y+7), (icon_x+14), (icon_y+7), 255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x+1),  (icon_y),   (icon_x+1),  (icon_y+7), 255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x),    (icon_y+2), (icon_x),    (icon_y+5), 255, 255, 255, 1);

    int32_t soc_bar_length = (int32_t)(10.0f * ((float)global_state->ups_soc / 100.0f));
    soc_bar_length = (soc_bar_length > 9) ? 9 : soc_bar_length;
    gfx_draw_line(global_state->gfx, (icon_x+12) - soc_bar_length, (icon_y+2), (icon_x+12), (icon_y+2), 255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x+12) - soc_bar_length, (icon_y+3), (icon_x+12), (icon_y+3), 255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x+12) - soc_bar_length, (icon_y+4), (icon_x+12), (icon_y+4), 255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x+12) - soc_bar_length, (icon_y+5), (icon_x+12), (icon_y+5), 255, 255, 255, 1);

    // 显示电量信息文字（若启用自动关机，在电量信息尾部追加关机倒计时 " | mm:ss"；
    // splash 每 100ms 重绘，倒计时随之逐秒刷新）
    wchar_t battery_info_buf[112];
    wchar_t auto_shutdown_buf[24] = L"";
    if (global_state->auto_shutdown_deadline != 0) {
        uint64_t remain_ms = (global_state->timestamp < global_state->auto_shutdown_deadline)
                           ? (global_state->auto_shutdown_deadline - global_state->timestamp) : 0;
        uint32_t remain_s = (uint32_t)((remain_ms + 999) / 1000); // 向上取整，避免剩余不足1秒时显示 00:00
        swprintf(auto_shutdown_buf, 24, L" | %02d:%02d", (int)(remain_s / 60), (int)(remain_s % 60));
    }
    swprintf(battery_info_buf, 112, L"电量:%d%% | %dmV | %dmA%ls%ls", global_state->ups_soc, global_state->ups_voltage, global_state->ups_current, (global_state->ups_is_charging ? L"  |  正在充电" : L""), auto_shutdown_buf);
    // 背景（阴影）
    gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, battery_info_buf, global_state->gfx->width/2 + 1, battery_center_y + 1, 0x66, 0x66, 0x66, 1);
    // 前景
    gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, battery_info_buf, global_state->gfx->width/2, battery_center_y, time_red, time_green, time_blue, 1);

#endif


#ifdef ASR_ENABLED
    // 检查ASR服务状态，如果ASR服务未启动，则在屏幕左上角画一个闪烁的点，表示ASR服务启动中
    if (global_state->is_asr_server_up < 1) {
        uint8_t v = (uint8_t)((global_state->timer >> 2) & 0x1);
        gfx_draw_line(global_state->gfx, 4, 6, 7, 6, 255, 255, 255, v);
        gfx_draw_line(global_state->gfx, 4, 7, 7, 7, 255, 255, 255, v);
        gfx_draw_line(global_state->gfx, 4, 8, 7, 8, 255, 255, 255, v);
        gfx_draw_line(global_state->gfx, 4, 9, 7, 9, 255, 255, 255, v);
    }
#endif

    gfx_refresh(global_state->gfx);
}






// ===============================================================================
// UI主体框架
// ===============================================================================

int32_t main_init(Key_Event *key_event, Global_State *global_state) {

    key_event->key_code = NANO_KEY_IDLE; // 大于等于16为没有任何按键，0-15为按键
    key_event->key_edge = 0;   // 0：松开  1：上升沿  -1：下降沿(短按结束)  -2：下降沿(长按结束)
    key_event->key_timer = 0;  // 按下计时器
    key_event->key_mask = 0;   // 长按超时后，键盘软复位标记。此时虽然物理上依然按键，只要软复位标记为1，则认为是无按键，无论是边沿还是按住都不触发。直到物理按键松开后，软复位标记清0。
    key_event->key_repeat = 0; // 触发一次长按后，只要不松手，该标记置1，直到物理按键松开后置0。若该标记为1，则在按住时触发连续重复动作。


    ///////////////////////////////////////
    // gfx初始化

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    
#else
    global_state->gfx = (Nano_GFX*)platform_calloc(1, sizeof(Nano_GFX));
    global_state->gfx->is_double_buffer = 0;
    gfx_init(global_state->gfx, SCREEN_WIDTH, SCREEN_HEIGHT, GFX_COLOR_MODE_RGB888);
#endif


    ui_init(key_event, global_state);

    ui_widget_textarea_init(key_event, global_state, global_state->w_textarea_main, UI_STR_BUF_MAX_LENGTH);




    ///////////////////////////////////////
    // UPS传感器初始化
#ifdef UPS_ENABLED
    ups_init();
#endif

    ///////////////////////////////////////
    // IMU初始化
#ifdef IMU_ENABLED
    imu_init();
    imu_calib();
#endif

    ///////////////////////////////////////
    // 输入设备初始化

    input_device_init();
    ui_softkbd_init(); // 触屏软键盘（内部初始化触屏HAL）
    ui_grid16kbd_init(); // 16键虚拟键盘
    key_event->prev_key = NANO_KEY_IDLE;

    ///////////////////////////////////////
    // 初始化玲珑天象仪（配置对象 + 专用画布，实现见 ui_linglong 模块）

    ui_linglong_global_init(global_state);


    ///////////////////////////////////////
    // 初始化文件系统

    fs_init();


    global_state->timezone = 8.0f;
    global_state->longitude = 119.0f;
    global_state->latitude = 32.0f;

    return 0;
}



// ===============================================================================
// 电源键锁屏（2026-08，仅 NANO_HAS_PWR_KEY 平台（M5Core2/M5CoreS3，PMIC PEK）编译）：
// 短按电源键（NANO_KEY_POWER，Core1 轮询 PMIC PEK 产事件）任意状态灭屏进锁屏、再短按恢复。
// 仅灭屏不睡 CPU（AXP IRQ 未接 GPIO，电源键无法唤醒，且灭屏断背光已是最大省电项）；
// RAM/帧缓冲/状态机原地保留，恢复=唤醒 LCD+重推帧缓冲。
// 有声/实时外设状态（占用 I2S/DMA）锁屏即走其 A 键退出路径先行退出，解锁后落在其上级菜单。
// ===============================================================================

// 锁屏激活判定：无 PMIC 电源键平台（NANO_HAS_PWR_KEY=0）恒假，锁屏相关抑制逻辑不参与
#if NANO_HAS_PWR_KEY
#define LOCK_SCREEN_ACTIVE(gs) ((gs)->STATE == STATE_LOCK_SCREEN)
#else
#define LOCK_SCREEN_ACTIVE(gs) (0)
#endif

#if NANO_HAS_PWR_KEY

#define LOCK_SCREEN_BLINK_PERIOD_MS (3000)  // LED 心跳周期（表示机器处于开机状态）
#define LOCK_SCREEN_BLINK_ON_MS     (10)    // 短促闪烁点亮时长
#define LOCK_SCREEN_BLINK_BRIGHTNESS (32)  // 心跳亮度（0~255，仅支持亮度的机型生效）

// 心跳颜色循环（红→黄→绿→青→蓝→紫→白）：仅 CoreS3 的 WS2812 灯带有真彩；
// Core2 为 PMIC 单色灯（颜色参数忽略），仅保留单色占位
#if defined(NANO_PLATFORM_M5CORES3)
static const int32_t s_lock_blink_colors[] = {
    MISC_LED_COLOR_RED, MISC_LED_COLOR_YELLOW, MISC_LED_COLOR_GREEN, MISC_LED_COLOR_CYAN,
    MISC_LED_COLOR_BLUE, MISC_LED_COLOR_PURPLE, MISC_LED_COLOR_WHITE
};
#else
static const int32_t s_lock_blink_colors[] = { MISC_LED_COLOR_GREEN };
#endif
#define LOCK_SCREEN_BLINK_COLOR_NUM (sizeof(s_lock_blink_colors) / sizeof(s_lock_blink_colors[0]))
static uint32_t s_lock_blink_color_idx = 0;

static int32_t  s_lock_resume_state   = STATE_MAIN_MENU; // 解锁后回落的状态
static int32_t  s_lock_redraw_on_exit = 0;   // 1=锁屏时发生了外设态退出，解锁需走目标态首帧重绘
// 下一次心跳时刻（对 global_state->timestamp）。注意必须是 uint64_t：
// get_timestamp_in_ms 返回 gettimeofday 墙钟毫秒（远超 2^32），曾用 uint32_t 截断导致
// 比较恒真、心跳每帧重触发、LED 常亮不闪（2026-08 CoreS3 实测故障）
static uint64_t s_lock_next_blink_ms  = 0;

// 有声/实时外设状态锁屏即退出：复用各模块 A 键（NANO_KEY_esc）退出路径（停 I2S/DMA/采集任务），
// 返回 1 表示发生了状态退出（解锁需走目标态首帧重绘，而非帧缓冲直恢）
static int32_t lock_screen_exit_peripheral_states(Key_Event *key_event, Global_State *global_state) {
    Key_Event fake = {0};
    fake.key_code = NANO_KEY_esc;
    fake.key_edge = -1;
    switch (global_state->STATE) {
        case STATE_MUSICBOX_PLAYING: // 停播+释放解码资源 → 文件列表
            global_state->STATE = ui_musicbox_playing_event(&fake, global_state);
            return 1;
        case STATE_OFDM_TXING:       // 中止发射 → 发射文本输入态
            global_state->STATE = ui_ofdm_txing_event(&fake, global_state);
            return 1;
        case STATE_OFDM_RX:          // 停采集任务+关麦 → 寻呼机模式菜单
            global_state->STATE = ui_ofdm_rx_event(&fake, global_state);
            return 1;
        case STATE_OFDM_MSG:         // 停接收/停发射 → 寻呼机模式菜单
            global_state->STATE = ui_ofdm_msg_event(&fake, global_state);
            return 1;
        default:
            return 0;
    }
}

static void lock_screen_enter(Key_Event *key_event, Global_State *global_state) {
    // 有声/实时外设状态先行退出（全局拦截点在状态机 switch 之前，此处直接改 STATE 安全）
    s_lock_redraw_on_exit = lock_screen_exit_peripheral_states(key_event, global_state);
    s_lock_resume_state = global_state->STATE;
    // 前台灵机 App：显式挂起（不再泵它、(App.is_active)=#f）。运行时与内存池原地保留，
    // 解锁后 resume 同一实例——不再走"伪 esc 硬退"那套。
    if (s_lock_resume_state == STATE_SCHEME_APP_RUN) ui_scheme_app_suspend_current();
    // 先置态：Core1 生产端检测到锁屏态随即停止按键/触屏事件投递与按键反馈
    global_state->STATE = STATE_LOCK_SCREEN;
    // 灭屏：面板背光置0（Core2 物理切断 AXP192 DCDC3 / CoreS3 切断 BLDO1）+ SLPIN 令 LCD 控制器睡眠。
    // 帧缓冲保持不动（RAM 中仍保留锁屏前画面），安静态解锁直接重推即可。
    display_sleep();
    s_lock_next_blink_ms = global_state->timestamp + LOCK_SCREEN_BLINK_PERIOD_MS;
    printf("lock screen: enter\n");
}

static void lock_screen_exit(Key_Event *key_event, Global_State *global_state) {
    display_wakeup(); // SLPOUT + 恢复睡眠前亮度
    sleep_in_ms(150); // ILI9342 睡眠退出恢复时间（约 120ms），期间推帧会被面板忽略
    display_set_brightness((uint8_t)global_state->brightness); // 显式恢复到设置值（与 wakeup 内部恢复一致，双保险）
    global_state->STATE = s_lock_resume_state;
    // 回到灵机 App：解除挂起（(App.resumed)=#t → App 补一次重绘）
    if (s_lock_resume_state == STATE_SCHEME_APP_RUN) ui_scheme_app_resume_current();
    if (!s_lock_redraw_on_exit) {
        // 安静态：抑制目标状态“首帧重初始化”（避免资源类状态重复初始化副作用），
        // 帧缓冲仍保留锁屏前画面，标脏全屏重推即恢复
        global_state->PREV_STATE = s_lock_resume_state;
        gfx_mark_dirty_full(global_state->gfx);
        gfx_refresh(global_state->gfx);
    }
    // 外设态退出情形：PREV_STATE 保持 LOCK_SCREEN，目标菜单态首帧重绘机制自动接管
    misc_led_set(0, MISC_LED_COLOR_GREEN, 255); // 心跳可能正处于点亮期，确保熄灭
    printf("lock screen: exit\n");
}

#endif // NANO_HAS_PWR_KEY


// ===============================================================================
// Animac终端：退出善后（共用）与退出确认模态框
// ===============================================================================

// 退出控制台善后（离开 STATE_ANIMAC_* 时调用；两条退出路径共用——
// 页眉“返回”经退出确认模态框确认后调用，控件 prev_focus_state 路径在 CONSOLE 分支尾部调用）：
// 销毁解释器上下文释放内存（约2MB PSRAM）、复位终端布局关联、恢复字体、日志区单例几何善后。
//（软键盘收起与布局恢复已由文本输入控件的退出路径固有处理，见 ui.c ui_widget_input_on_leave）
static void ui_app_animac_cleanup(Key_Event *key_event, Global_State *global_state) {
    // 控件固有善后（输入法状态机/候选/倒计时/输入模式/全局Ctrl全复位 + 双键盘收起 + 布局恢复）——
    // 原“返回”经输入控件页眉热点路径固有调用，模态框拦截后须显式补齐，否则键盘可见性/
    // Ctrl/输入法状态残留会扭曲后续所有状态的布局与触屏热点
    ui_widget_input_back_cleanup(global_state, global_state->w_input_main);
    ui_animac_close(key_event, global_state);
    global_state->w_input_main->dyn_height = 0;
    global_state->w_input_main->log_view = NULL;
    global_state->w_textarea_main->is_bare = 0;
    global_state->ui_font = s_animac_prev_ui_font; // 恢复进入 STATE_ANIMAC_* 之前的字体
    // 日志区（w_textarea_main 全局单例）善后：先恢复字体，再按恢复后的字体复位
    // 标准布局几何与滚动位置——否则控制台终端修改过的 y/height 会遗留给下一个使用者
    ui_widget_textarea_reset_geometry(key_event, global_state, global_state->w_textarea_main);
}

// ===============================================================================
// 安全关机：手机式「滑动关机」滑块
//
// 交互：抓住轨道左端的圆形推钮向右拖动，滑到右端（进度 ≥ UI_SHUTDOWN_COMPLETE_PCT）
// 后松手才真正调用 graceful_shutdown()；未滑到位松手则回弹到起点。触屏只认推钮
// （含少量触摸余量），点按轨道其余位置不会误触发。触屏上不派生任何键码
//（全屏宫格软键已整体退场），触屏取消入口是页眉右侧「返回」热点；
// 实体键盘保留原有快捷方式：长按 D 关机、A 取消。
//
// 本状态直接消费触屏流（DOWN 锁存推钮、拖动逐帧跟随、UP 沿结算），符合 ui_app.c
// 输入处理原则：会引发状态切换的动作只在松手沿触发。
//
// 绘制几何（严丝合缝）：轨道为胶囊（圆角半径 = 高度/2）；已滑过部分是与轨道同高、
// 同一圆角半径的胶囊，右端半圆即推钮（半径 = 轨道高度/2），左端与轨道左端完全重合；
// 进度 0 时填充宽 = 轨道高 = 推钮直径（正圆），100% 时恰为轨道全宽。整条滑轨只有
// 两种几何形状（轨道胶囊 / 推钮胶囊）且共用同一半径，不存在接缝或半径不一致。
// 全部图形都走抗锯齿原语：圆角矩形（8×8 超采样）与抗锯齿线段（箭头）。
// ===============================================================================

#define UI_SHUTDOWN_TRACK_MAX_W   (260) // 轨道最大宽度（大屏居中，不铺满全屏）
#define UI_SHUTDOWN_TRACK_MAX_H   (48)  // 轨道最大高度（胶囊高度）
#define UI_SHUTDOWN_TRACK_MIN_H   (20)  // 轨道最小高度（128x64 小屏仍可拖动）
#define UI_SHUTDOWN_COMPLETE_PCT  (90)  // 松手判定阈值：滑过 90% 即视为滑到底
#define UI_SHUTDOWN_GRAB_SLOP     (12)  // 推钮命中额外余量（px），便于手指抓取

typedef struct Shutdown_Ui {
    int32_t track_x, track_y, track_w, track_h; // 轨道矩形（胶囊）
    int32_t knob_r;        // 推钮半径（恒 = track_h / 2，与轨道端部同一半径）
    int32_t knob_x0, knob_x1; // 推钮圆心可移动范围（左端 / 右端）
} Shutdown_Ui;

typedef struct Shutdown_Palette {
    uint8_t track[3];     // 轨道底色
    uint8_t accent[3];    // 已滑过部分 / 推钮（同一胶囊）
    uint8_t hot[3];       // 滑到位（可松手）时的强调色
    uint8_t label[3];     // 轨道内提示文字（未滑过处）
    uint8_t on_accent[3]; // 压于强调色之上的文字 / 箭头
} Shutdown_Palette;

static int32_t s_shutdown_progress = 0;      // 滑块进度 0..100
static int32_t s_shutdown_dragging = 0;      // 是否正抓着推钮拖动
static int32_t s_shutdown_grab_dx = 0;       // 抓取点与推钮圆心的水平偏差（跟手不跳变）
static int32_t s_shutdown_drawn_progress = -1; // 上次绘制的进度（仅在变化时重绘）
static int32_t s_shutdown_busy = 0;          // 正在关机 / 失败提示中，忽略输入

static void ui_shutdown_layout(Global_State *global_state, Shutdown_Ui *u) {
    const int32_t w = (int32_t)global_state->gfx->width;
    const int32_t h = (int32_t)global_state->gfx->height;
    const int32_t header_h = ui_std_header_height(global_state->ui_font);
    const int32_t footer_h = gfx_font_line_height(global_state->ui_font) + 1;

    // 轨道宽度：左右各留 10% 边距，大屏封顶；极窄屏退化为贴边
    int32_t track_w = w - 2 * (w / 10);
    if (track_w > UI_SHUTDOWN_TRACK_MAX_W) track_w = UI_SHUTDOWN_TRACK_MAX_W;
    if (track_w < 64) track_w = w - 8;
    if (track_w < 16) track_w = w;

    // 轨道高度：可用高度（页眉与页脚之间）的 1/4，钳制到 [MIN, MAX]
    int32_t avail_h = h - header_h - footer_h;
    if (avail_h < 16) avail_h = 16;
    int32_t track_h = avail_h / 4;
    if (track_h > UI_SHUTDOWN_TRACK_MAX_H) track_h = UI_SHUTDOWN_TRACK_MAX_H;
    if (track_h < UI_SHUTDOWN_TRACK_MIN_H) track_h = UI_SHUTDOWN_TRACK_MIN_H;
    if (track_h > avail_h - 4) track_h = avail_h - 4;
    if (track_h < 12) track_h = 12;

    u->track_w = track_w;
    u->track_h = track_h;
    u->track_x = (w - track_w) / 2;
    u->track_y = header_h + (avail_h - track_h) / 2;
    if (u->track_y < header_h) u->track_y = header_h;

    // 推钮 = 轨道端部半圆：半径恒为轨道高度的一半，圆心在轨道内缘（不内缩），
    // 因此推钮轮廓与胶囊端部完全重合，进度 0/100% 时严丝合缝地贴住轨道两端
    u->knob_r = track_h / 2;
    if (u->knob_r < 6) u->knob_r = 6;
    u->knob_x0 = u->track_x + u->knob_r;
    u->knob_x1 = u->track_x + track_w - u->knob_r;
    if (u->knob_x1 < u->knob_x0) u->knob_x1 = u->knob_x0;
}

static int32_t ui_shutdown_knob_cx(const Shutdown_Ui *u, int32_t progress) {
    int32_t span = u->knob_x1 - u->knob_x0;
    if (progress < 0) progress = 0;
    if (progress > 100) progress = 100;
    return u->knob_x0 + span * progress / 100;
}

static int32_t ui_shutdown_progress_from_x(const Shutdown_Ui *u, int32_t x) {
    int32_t span = u->knob_x1 - u->knob_x0;
    if (span <= 0) return 0;
    int32_t p = ((x - u->knob_x0) * 100 + span / 2) / span;
    if (p < 0) p = 0;
    if (p > 100) p = 100;
    return p;
}

static void ui_shutdown_palette(Global_State *global_state, Shutdown_Palette *p) {
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        p->track[0] = 230; p->track[1] = 232; p->track[2] = 237;
        p->accent[0] = 214; p->accent[1] = 52; p->accent[2] = 56;
        p->hot[0] = 238; p->hot[1] = 78; p->hot[2] = 58;
        p->label[0] = 126; p->label[1] = 131; p->label[2] = 139;
        p->on_accent[0] = 255; p->on_accent[1] = 255; p->on_accent[2] = 255;
    }
    else {
        p->track[0] = 56; p->track[1] = 58; p->track[2] = 64;
        p->accent[0] = 212; p->accent[1] = 56; p->accent[2] = 60;
        p->hot[0] = 240; p->hot[1] = 88; p->hot[2] = 60;
        p->label[0] = 176; p->label[1] = 180; p->label[2] = 188;
        p->on_accent[0] = 255; p->on_accent[1] = 255; p->on_accent[2] = 255;
    }
}

// 绘制滑块（轨道 + 已滑过部分 + 提示文字 + 推钮）：整块轨道区域每次全量重绘，
// 旧推钮痕迹被轨道底色覆盖；脏区自动收敛为轨道包围盒，拖动时只推这一块。
//
// 层序：轨道 → 暗色文字 → 填充胶囊（含推钮）→ 填充区内亮色文字（裁到推钮左缘）→ 箭头。
// 文字“被推钮推走”的揭示边界正是推钮左缘，与填充胶囊的端部完全对齐。
static void ui_shutdown_draw_slider(Global_State *global_state, const Shutdown_Ui *u, int32_t progress) {
    Nano_GFX *gfx = global_state->gfx;
    Shutdown_Palette p;
    ui_shutdown_palette(global_state, &p);

    const int32_t x = u->track_x;
    const int32_t y = u->track_y;
    const int32_t w = u->track_w;
    const int32_t h = u->track_h;
    const int32_t r = h / 2;                 // 轨道 / 填充 / 推钮共用同一圆角半径
    const int32_t cy = y + r;
    const int32_t knob_cx = ui_shutdown_knob_cx(u, progress);
    const int32_t fill_w = knob_cx - x + r;  // 右端半圆（圆心 knob_cx）即推钮，故 +r
    const int32_t armed = (progress >= UI_SHUTDOWN_COMPLETE_PCT);
    const uint8_t *ac = armed ? p.hot : p.accent;
    const uint32_t label_font = (h >= 36) ? GFX_FONT_ALPHA_16 : GFX_FONT_ALPHA_12;
    const int32_t text_cx = x + w / 2;

    // 1. 轨道胶囊（抗锯齿圆角矩形；圆角半径 = 高度/2）
    gfx_draw_rounded_rectangle(gfx, x, y, w, h, r, r, r, r,
        p.track[0], p.track[1], p.track[2], 1);

    // 2. 轨道内提示文字（未滑过配色）；滑过部分稍后会被填充完整盖住
    gfx_font_draw_text_centered(gfx, label_font, L"滑动关机", text_cx, cy,
        p.label[0], p.label[1], p.label[2], 1);

    // 3. 填充胶囊 = 已滑过部分 + 推钮（同一个图形，无接缝）：
    //    fill_w 恒 ≥ 2r——进度 0 时恰为直径 h 的正圆（推钮停在左端），100% 时恰等于轨道全宽
    if (fill_w > 0) {
        gfx_draw_rounded_rectangle(gfx, x, y, fill_w, h, r, r, r, r,
            ac[0], ac[1], ac[2], 1);
    }

    // 4. 填充区内的亮色文字：裁剪到推钮左缘，形成“文字被推走”的揭示效果
    {
        int32_t clip_w = (knob_cx - r) - x;
        if (clip_w > 0) {
            gfx_set_clip(gfx, x, y, clip_w, h);
            gfx_font_draw_text_centered(gfx, label_font, L"滑动关机", text_cx, cy,
                p.on_accent[0], p.on_accent[1], p.on_accent[2], 1);
            gfx_reset_clip(gfx);
        }
    }

    // 5. 推钮内的右向箭头：抗锯齿线段（加色模式 3：白线在红色胶囊上自然过渡）
    {
        int32_t s = r / 2;                   // 箭头半高
        if (s < 4) s = 4;
        const float ax = (float)knob_cx + (float)s * 0.15f; // 视觉居中略偏右一点点
        const float ay = (float)cy;
        const float lw = (r >= 18) ? 3.0f : 2.0f;
        gfx_draw_line_anti_aliasing(gfx, ax - (float)s * 0.5f, ay - (float)s,
            ax + (float)s * 0.5f, ay, lw,
            p.on_accent[0], p.on_accent[1], p.on_accent[2], 3);
        gfx_draw_line_anti_aliasing(gfx, ax + (float)s * 0.5f, ay,
            ax - (float)s * 0.5f, ay + (float)s, lw,
            p.on_accent[0], p.on_accent[1], p.on_accent[2], 3);
    }
}

// 页眉高度（整页与提示页共用；页眉右侧「返回」热点也以此判定）
static int32_t ui_shutdown_header_h(Global_State *global_state) {
    return ui_std_header_height(global_state->ui_font);
}

// 整页重绘（清屏 + 页眉 / 页脚 + 滑块）。页眉右侧固定「返回」热点：触屏设备
// （无硬键盘）在关机页的取消入口，避免进入后无法退出（触屏不派生键码）。
static void ui_shutdown_draw_page(Key_Event *key_event, Global_State *global_state) {
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_fill_white(global_state->gfx);
    }
    else {
        gfx_soft_clear(global_state->gfx);
    }
    ui_draw_header_full(key_event, global_state, L"安全关机", 1,
        ui_shutdown_header_h(global_state), NULL, L"返回 ");
#if NANO_HAS_HW_KEYBOARD
    ui_draw_footer(key_event, global_state, L"长按D键关机  短按A键取消", 1);
#else
    ui_draw_footer(key_event, global_state, L"滑到最右端即可关机", 1);
#endif
    Shutdown_Ui u;
    ui_shutdown_layout(global_state, &u);
    ui_shutdown_draw_slider(global_state, &u, s_shutdown_progress);
    s_shutdown_drawn_progress = s_shutdown_progress;
}

// 关机过程 / 失败提示（全屏居中一行）
static void ui_shutdown_draw_status(Key_Event *key_event, Global_State *global_state,
                                    const wchar_t *text, int32_t is_error) {
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_fill_white(global_state->gfx);
    }
    else {
        gfx_soft_clear(global_state->gfx);
    }
    ui_draw_header(key_event, global_state, L"安全关机", 1);
    uint8_t tr = 240, tg = 240, tb = 240;
    if (is_error) { tr = 220; tg = 64; tb = 64; }
    else if (global_state->ui_color_style == UI_COLOR_LIGHT) { tr = 34; tg = 34; tb = 34; }
    gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_16, (wchar_t *)text,
        (int32_t)global_state->gfx->width / 2, (int32_t)global_state->gfx->height / 2, tr, tg, tb, 1);
    gfx_refresh(global_state->gfx);
}

// 真正执行关机（滑块滑到位松手 / 长按 D 两条入口共用）
static void ui_shutdown_do_shutdown(Key_Event *key_event, Global_State *global_state) {
    s_shutdown_busy = 1;
    s_shutdown_dragging = 0;

    ui_shutdown_draw_status(key_event, global_state, L"正在安全关机...", 0);

    // 成功时本函数不返回（ESP32 已断电）；Linux 的 poweroff 成功返回 0 也不回到
    // 失败分支，保持“正在安全关机...”画面并忽略输入，与原实现语义一致。
    if (graceful_shutdown() < 0) {
        // 关机失败：提示后返回主菜单
        ui_shutdown_draw_status(key_event, global_state, L"安全关机失败", 1);
        sleep_in_ms(1000);
        s_shutdown_busy = 0;
        global_state->STATE = STATE_MAIN_MENU;
    }
}


int32_t main_event_handler(Key_Event *key_event, Global_State *global_state) {

    // 将时间戳转为本地日期时间
    time_t ts = (time_t)(global_state->timestamp / 1000);
    struct tm *timeinfo = localtime(&ts);
    global_state->year = timeinfo->tm_year + 1900;
    global_state->month = timeinfo->tm_mon + 1;
    global_state->day = timeinfo->tm_mday;
    global_state->hour = timeinfo->tm_hour;
    global_state->minute = timeinfo->tm_min;
    global_state->second = timeinfo->tm_sec;
    global_state->millisecond = global_state->timestamp % 1000;


    // 电源键全局拦截（任何状态，仅认下降沿）：非锁屏态 → 锁屏；锁屏态 → 解锁。
    // 处理完直接返回，本帧不再进入状态机分支。
#if NANO_HAS_PWR_KEY
    if (key_event->key_code == NANO_KEY_POWER && key_event->key_edge == -1) {
        if (global_state->STATE == STATE_LOCK_SCREEN) lock_screen_exit(key_event, global_state);
        else                                          lock_screen_enter(key_event, global_state);
        return 0;
    }
#endif

    // 主状态机
    switch(global_state->STATE) {

    /////////////////////////////////////////////
    // 初始状态：欢迎屏幕。按任意键进入主菜单
    /////////////////////////////////////////////

    case STATE_SPLASH_SCREEN:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {

        }
        global_state->PREV_STATE = global_state->STATE;

        if (global_state->timestamp - last_splash_timestamp >= 100) {
            ui_app_splash_render_frame(key_event, global_state);
            last_splash_timestamp = global_state->timestamp;
        }

        // 进入主菜单：实体键盘任意键（不论长短按），或触屏松手沿（任意位置点按）。
        // 触屏入口必须有：全屏 4x4 宫格隐式软键退场后，纯触屏设备上没有别的键可按。
        if (key_event->key_edge < 0 && key_event->key_code != NANO_KEY_IDLE) {
            global_state->STATE = STATE_MAIN_MENU;
        }
        else if (key_event->touch_edge & TOUCH_EDGE_UP) {
            global_state->STATE = STATE_MAIN_MENU;
        }

        break;

    /////////////////////////////////////////////
    // 主菜单。
    /////////////////////////////////////////////

    case STATE_MAIN_MENU:

        // 首次获得焦点：重建条目表（内置项 + app/ 目录扫描出的灵机 App，含运行角标）
        // 并整屏重绘（左滚动区 + 右固定区，内含推屏）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_main_menu_entry(global_state);
            ui_main_menu_draw(key_event, global_state, 1);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_main_menu_event_handler(key_event, global_state);

        break;

    /////////////////////////////////////////////
    // 电子书：文件列表菜单
    /////////////////////////////////////////////

    case STATE_EBOOK:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            // 统一为“页眉页脚先入帧缓冲、菜单绘制最后统一刷屏”，避免两阶段断续感
            ui_draw_header(key_event, global_state, (wchar_t *)global_state->w_menu_main->title, 1);
            ui_widget_menu_refresh(key_event, global_state, global_state->w_menu_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_widget_menu_event_handler(key_event, global_state, global_state->w_menu_main, ui_ebook_menu_item_action, STATE_MAIN_MENU, STATE_EBOOK);

        break;

    /////////////////////////////////////////////
    // 电子书：阅读
    /////////////////////////////////////////////

    case STATE_EBOOK_READING:

        // 首次获得焦点：渲染当前页
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ebook_reading_render(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 翻页/跳页/返回等按键处理（发生变化的按键内部自行触发重绘）
        ui_ebook_reading_event_handler(key_event, global_state);

        break;

    /////////////////////////////////////////////
    // 文字编辑器状态
    /////////////////////////////////////////////

    case STATE_LLM_INPUT:

        global_state->STATE = ui_llm_input_event_handler(key_event, global_state);

        break;

    /////////////////////////////////////////////
    // 选择语言模型状态
    /////////////////////////////////////////////

    case STATE_MODEL_MENU:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_draw_header(key_event, global_state, (wchar_t *)global_state->w_menu_main->title, 1);
            ui_widget_menu_refresh(key_event, global_state, global_state->w_menu_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_widget_menu_event_handler(key_event, global_state, global_state->w_menu_main, model_menu_item_action, STATE_MAIN_MENU, STATE_MODEL_MENU);

        // 退出模型菜单回到主菜单时，卸载当前模型（释放 infer.c 上下文与小鹦鹉笼引擎）；
        // 再次进入时会经“选模型”流程重新建立
        if (global_state->STATE == STATE_MAIN_MENU) {
            ui_llm_unload_model(global_state);
        }

        break;


    /////////////////////////////////////////////
    // 小游戏菜单
    /////////////////////////////////////////////

    case STATE_GAME_MENU:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            // 从主菜单或灵机应用程序返回时，都重建游戏菜单条目；
            // 否则状态已切换，w_menu_main 仍保留上一菜单的标题与内容。
            init_game_menu(key_event, global_state);
            // 统一为“页眉页脚先入帧缓冲、菜单绘制最后统一刷屏”，避免两阶段断续感
            ui_draw_header(key_event, global_state, (wchar_t *)global_state->w_menu_main->title, 1);
            ui_widget_menu_refresh(key_event, global_state, global_state->w_menu_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_widget_menu_event_handler(key_event, global_state, global_state->w_menu_main, game_menu_item_action, STATE_MAIN_MENU, STATE_GAME_MENU);

        break;


    /////////////////////////////////////////////
    // 设置菜单
    /////////////////////////////////////////////

    case STATE_SETTING_MENU:

        // 首次获得焦点：初始化并绘制。从编辑器返回时保留列表滚动位置。
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_settings_menu_enter(global_state, global_state->PREV_STATE != STATE_SETTING_INPUT);
            ui_settings_menu_draw(global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_settings_menu_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 语言推理进行中（异步，每个iter结束后会将控制权交还事件循环，而非自行阻塞到最后一个token）
    //   实际上就是将generate_sync的while循环打开，将其置于大的事件循环。
    /////////////////////////////////////////////

    case STATE_LLM_ON_INFER:

        global_state->STATE = ui_llm_on_infer_event_handler(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 推理结束（自然结束或中断），显示推理结果
    /////////////////////////////////////////////

    case STATE_LLM_AFTER_INFER:

        global_state->STATE = ui_llm_after_infer_event_handler(key_event, global_state);

        break;

    /////////////////////////////////////////////
    // ASR实时识别进行中（响应ASR客户端回报的ASR文本内容）
    /////////////////////////////////////////////

    case STATE_ASR_RUNNING:
#ifdef ASR_ENABLED
        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            // 设置PTT状态为按下（>0）
            if (set_ptt_status(66) < 0) break;

            // 打开ASR管道
            if (open_asr_fifo() < 0) break;


            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L"请说话...", 0, 0);

            global_state->is_recording = 1;
            global_state->asr_start_timestamp = global_state->timestamp;
        }
        global_state->PREV_STATE = global_state->STATE;

        // 实时显示ASR结果
        if (global_state->is_recording == 1) {
            int32_t len = read_asr_fifo(global_state->asr_output_buffer);
            (void)len;

            // 临时关闭draw_textarea的整帧绘制，以便在textarea上绘制进度条之后再统一写入屏幕，否则反复的clear会导致进度条闪烁。
            global_state->is_full_refresh = 0;
            gfx_soft_clear(global_state->gfx);

            // 显示ASR结果
            // if (len > 0) {
                ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, global_state->asr_output_buffer, -1, 1);
                ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
            // }

            // 绘制录音持续时间（位于底栏位置，底栏高度跟随当前字体行高）
            wchar_t rec_duration[50];
            swprintf(rec_duration, 50, L" %ds ", (uint32_t)((global_state->timestamp - global_state->asr_start_timestamp) / 1000));
            gfx_draw_textline(global_state->gfx, rec_duration, 0, global_state->gfx->height - (gfx_font_line_height(global_state->ui_font) + 1), 255, 255, 255, 0);

            gfx_refresh(global_state->gfx);

            // 重新开启整帧绘制，注意这个标记是所有函数共享的全局标记。
            global_state->is_full_refresh = 1;

        }

        // 松开按钮，停止PTT
        if (global_state->is_recording > 0 && key_event->key_edge == 0 && key_event->key_code == NANO_KEY_IDLE) {

            global_state->is_recording = 0;
            global_state->asr_start_timestamp = 0;

            close_asr_fifo();

            // // 设置PTT状态为松开（==0）
            if (set_ptt_status(0) < 0) break;
            close_ptt_fifo();

            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L" \n \n      识别完成", 0, 0);
            ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);

            sleep_in_ms(500);

            wcscpy(global_state->w_input_main->textarea.text, global_state->asr_output_buffer);
            global_state->w_input_main->textarea.length = wcslen(global_state->asr_output_buffer);

            wcscpy(global_state->asr_output_buffer, L"请说话...");

            // ASR后立刻提交到LLM？
            if (global_state->is_auto_submit_after_asr) {
                global_state->STATE = STATE_LLM_ON_INFER;
            }
            else {
                global_state->w_input_main->current_page = 0;
                global_state->STATE = STATE_LLM_INPUT;
            }

        }

        // 短按A键：清屏，清除输入缓冲区，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_esc) {
            // 刷新文本输入框
            ui_widget_input_init(key_event, global_state, global_state->w_input_main, global_state->llm_model_name);
            global_state->STATE = STATE_LLM_INPUT;
        }
#endif
        break;

    /////////////////////////////////////////////
    // 本机自述
    /////////////////////////////////////////////

    case STATE_README: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_draw_header(key_event, global_state, L"本机自述", 1);
            ui_draw_footer(key_event, global_state, L"(c) 2025-2026 BD4SUR", 1);

            wchar_t readme[1024];
            wchar_t color_reset_tag[10];
            if (global_state->ui_color_style == UI_COLOR_LIGHT) {
                wcscpy(color_reset_tag, L"[#000000]");
            }
            else if (global_state->ui_color_style == UI_COLOR_DARK) {
                wcscpy(color_reset_tag, L"[#ffffff]");
            }
            swprintf(readme, 1024, L"[#66ccff]LinglongOS%ls V" NANO_VERSION "\n玲珑操作系统 (WASM)\n版权所有 (c) 2018-2026 BD4SUR\nhttps://bd4sur.com\n\n# 概述\n\n玲珑操作系统(LLOS)是BD4SUR开发的面向便携计算设备的嵌入式操作系统。\n\nLLOS由灵机计算引擎(Animac)驱动。用户能够自由开发、运行Scheme或JavaScript编写的应用程序。\n\nLLOS机制透明、接口清晰，能够与智能代理充分互动互操作，因此任何用户都可以借助HI“许愿式”制作自己的App。\n\nLLOS能够运行在几乎所有硬件上，为简单设备赋予灵活的二次开发能力。\n\n# 研制历史\n\nLLOS最早可追溯到BD4SUR从2018年开始开发的Scheme计算引擎“灵机”。灵机引擎为LLOS赋予开放的应用编程能力。\n\nLLOS的直接原型是2025年5月开始开发的自制LLM端侧推理演示设备，基于12864点阵屏幕和16键矩阵键盘，初步开发了文本显示、九键文本输入等基础功能，运行在路由器、树莓派、ESP32微控制器等设备上。\n\n2026年1月起，开始研制玲珑天象仪，针对彩色触摸屏幕重新设计UI。经过手工重构，形成了基本可用的嵌入式GUI框架。该框架将LLM推理、玲珑天象仪等多种功能，集成到统一的框架中，烧录进M5CoreS3等便携设备上，制成实用好玩的“电子核桃”。\n\n2026年3季度，将灵机计算引擎与电子核桃GUI框架集成一体，形成如今的LLOS形态。与此同时，将大部分应用功能从框架中剥离，用HI改写为Scheme，从固化到框架的预置功能变为可插拔、可定制的应用。\n\n# 研制团队\n\n产品定义与项目管理：\n· BD4SUR全体操作员\n软件编码：\n· BD4SUR\n· DeepSeek-V4.1-Flash (本地)\n· GLM5.3-Flash (本地)\n· Kimi K3 及其前代模型\n\n", color_reset_tag);

            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, readme, 0, 1);
            ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按A键返回主菜单（释放演化算法占用的 PSRAM）
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            ui_app_genetic_exit();
            global_state->STATE = STATE_MAIN_MENU;
        }

        global_state->STATE = ui_widget_textarea_event_handler(key_event, global_state, global_state->w_textarea_main, STATE_MAIN_MENU, STATE_README);

        break;
    }

    /////////////////////////////////////////////
    // 玲珑天象仪：计算太阳和月亮位置
    /////////////////////////////////////////////

    case STATE_LINGLONG:

        // 首次获得焦点：初始化后立即渲染首帧，不再有启动画面的有意停留。
        // 历史：此处曾用 sleep_in_ms(1000) 阻塞停留，后改为非阻塞跨帧停留 1s（因为 WASM-GUI
        // 下阻塞 sleep 是空实现、且启动画面与首帧会被压进同一次回调而看不到）；现按需求
        // 把这段延时整个去掉——进玲珑仪后第一帧就是星图/海面本身。
        // 首帧需要烘焙大气 LUT 与波场（WASM 下单线程约 1s），这段时间宿主仍显示上一界面。
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_linglong_init(key_event, global_state);
            global_state->PREV_STATE = global_state->STATE;
        }

        ui_linglong_event_handler(key_event, global_state);
        ui_linglong_render_frame(key_event, global_state);

        break;

    /////////////////////////////////////////////
    // FLIP流体模拟
    /////////////////////////////////////////////

    case STATE_FLIP:

        // 番茄表/电子沙漏：界面与交互整体在 ui_flip 模块（flip.c 提供流体求解与渲染）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_flip_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_flip_render_frame(key_event, global_state);
        ui_flip_event_handler(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 水波
    /////////////////////////////////////////////

    case STATE_RIPPLE: {

        // 首次获得焦点：初始化（从SD卡读取 /wallpaper/wp.png 并解码为纹理）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ripple_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按键处理（A返回）；返回则释放纹理与波场缓冲并跳过一帧渲染
        ui_ripple_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_RIPPLE) {
            ui_ripple_on_exit();
            break;
        }

        // 触摸激发 + 逻辑更新 + 渲染一帧
        ui_ripple_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 灵机应用程序：列表（动态扫描业务资源目录 app/ 下的 .scm）
    /////////////////////////////////////////////

    case STATE_SCHEME_APP_LIST:

        // 首次获得焦点：重新扫描目录并构建菜单（支持 App 热插拔）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_scheme_app_list_init(key_event, global_state);
            ui_draw_header(key_event, global_state, (wchar_t *)global_state->w_menu_main->title, 1);
            ui_widget_menu_refresh(key_event, global_state, global_state->w_menu_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 菜单事件（上下键/触屏/A返回游戏菜单；enter 经 item_action 转入运行态）。
        // 走模块自己的包装：它在菜单控件外面加了一层「退出 App 后在途触摸序列余波」
        // 抑制（见 ui_scheme_app.c 的 s_list_touch_guard），避免"确认退出"这一下
        // 被列表当成一次点击、直接又进入另一个 App。
        global_state->STATE = ui_scheme_app_list_event_handler(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 灵机应用程序：运行（解释执行选中的 .scm）
    /////////////////////////////////////////////

    case STATE_SCHEME_APP_RUN: {

        // 首次获得焦点：创建解释器运行时并加载选中的 Scheme 进程
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_scheme_app_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按键处理（硬 A/ESC 只置“返回请求”，不再在这里退出）；
        // 这个提前返回口现在理论上不可达（ui_scheme_app_event_handler 不再改 STATE），
        // 保留作防御：将来若事件层又要直接切状态，仍然会经过同一套实例收尾。
        ui_scheme_app_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_SCHEME_APP_RUN) {
            ui_scheme_app_on_exit();
            break;
        }

        // 泵一个时间片（解释器内部抢占分帧）；推屏时机由 App 源码控制
        ui_scheme_app_render_frame(key_event, global_state);

        // 退出 / 切后台 / 拉起判定，放在本帧泵完**之后**——这样 (App.exit) 返回 #t 之后，
        // App 还有整整一个时间片把当前这一拍里的收尾代码（落盘/清理）跑完，然后本帧末
        // 才销毁（退出）、冻结（切后台）或切换（拉起另一个 App）。
        // 优先级：退出 > 拉起 > 切后台（一个 App 同时发多条请求时行为确定）。
        if (ui_scheme_app_should_exit(global_state)) {
            // 所有退出的唯一出口：有活着的“返回链”（本 App 是被另一个 App 拉起的）就
            // 回上一 App 并**留在运行态**（与 perform_launch 同一手法：PREV_STATE 未变，
            // 下一帧不会再跑 init）；否则销毁实例回**主菜单**（2026-11：灵机 App 的图标
            // 现在就摆在主菜单里，退出即回主菜单，与“切后台”同一落点）。
            if (!ui_scheme_app_perform_exit(global_state)) {
                global_state->STATE = STATE_MAIN_MENU;
            }
        }
        else if (ui_scheme_app_perform_launch(global_state)) {
            // (App.launch)/(App.launch_with)：宿主机已完成切换（目标已上前台，
            // 调用者已冻结保留或销毁），状态机仍是 STATE_SCHEME_APP_RUN，
            // 下一帧由新 App 自己绘制（PREV_STATE 未变，不会再跑 init）。
        }
        else if (ui_scheme_app_should_hide(global_state)) {
            // (App.hide) / 退出确认框的「后台」：实例与内存池保留，回主菜单
            // （该 App 的图标带绿色运行角标，点它即恢复同一实例）
            ui_scheme_app_suspend_to_list();
            global_state->STATE = STATE_MAIN_MENU;
        }

        break;
    }


    /////////////////////////////////////////////
    // 水池（WebGL Water 移植）
    /////////////////////////////////////////////

    case STATE_WATER: {

        // 首次获得焦点：初始化（分配高度/速度场、法线场、视野映射与输出缓冲）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_water_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 事件处理（屏幕控件/拖动/实体键盘）；返回则释放全部内存并跳过一帧渲染
        ui_water_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_WATER) {
            ui_water_on_exit();
            break;
        }

        // 模拟 + 渲染一帧（拖动交互在 render_frame 内完成）
        ui_water_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 体积云与天空仿真
    /////////////////////////////////////////////

    case STATE_CLOUD: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_cloud_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按键处理（方向键视角 / 回车太阳 / 返回主菜单）；返回主菜单则跳过一帧渲染
        ui_cloud_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_CLOUD) break;

        // 渲染一帧
        ui_cloud_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 寻呼机（OFDM 声波数传）：模式菜单（发射/接收，同一时刻仅一种模式）
    /////////////////////////////////////////////

    case STATE_OFDM_MENU:

        // 首次获得焦点：初始化码表并绘制三张功能卡（发送/接收/环路自测）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ofdm_menu_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_ofdm_menu_event(key_event, global_state);

        // 退出寻呼机模块（回主菜单）：释放 modem 码表（PSRAM）
        if (global_state->STATE == STATE_MAIN_MENU) {
            ui_ofdm_menu_on_exit();
        }

        break;


    /////////////////////////////////////////////
    // 寻呼机：发射文本输入（复用全局单例输入控件，D 提交进入发射态）
    /////////////////////////////////////////////

    case STATE_OFDM_TX:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ofdm_tx_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_ofdm_tx_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 寻呼机：发射中（OFDM 音频经扬声器多通道循环播放，直到手动停止）
    /////////////////////////////////////////////

    case STATE_OFDM_TXING:

        // 首次获得焦点：调制并启动循环播放
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ofdm_txing_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_ofdm_txing_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 寻呼机：接收中（硅麦采集 → OFDM 解调 → 文本滚动显示）
    /////////////////////////////////////////////

    case STATE_OFDM_RX:

        // 首次获得焦点：接管麦克风并创建接收机
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ofdm_rx_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_ofdm_rx_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 寻呼机：软件环路自测文本输入（复用全局单例输入控件，D 提交进入环回态）
    /////////////////////////////////////////////

    case STATE_OFDM_LOOP:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ofdm_loop_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_ofdm_loop_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 寻呼机：软件环路自测中（发射机逐帧渲染 → 直接喂本机接收机，不出声）
    /////////////////////////////////////////////

    case STATE_OFDM_LOOPING:

        // 首次获得焦点：创建收发机并启动环回
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ofdm_looping_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_ofdm_looping_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 寻呼机：消息收发（一屏完成收发：记录区 + 接收/发射按钮 + 状态栏 + 发射进度条）
    /////////////////////////////////////////////

    case STATE_OFDM_MSG:

        // 首次获得焦点：绘制页面；若刚从发报输入页带回报文则立即开始连发 3 遍
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ofdm_msg_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_ofdm_msg_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 寻呼机：消息收发的发报输入（复用全局单例输入控件，D 提交 → 回消息页发射）
    /////////////////////////////////////////////

    case STATE_OFDM_MSG_INPUT:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ofdm_msg_input_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_ofdm_msg_input_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 音乐盒：文件列表（SD 卡根目录 WAV/MP3，复用全局单例菜单控件）
    /////////////////////////////////////////////

    case STATE_MUSICBOX_MENU:

        // 首次获得焦点：初始化（从主菜单进入时列表已由 ui_musicbox_menu_init 构建；
        // 从播放态返回时列表保留，仅重绘）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_draw_header(key_event, global_state, (wchar_t *)global_state->w_menu_main->title, 1);
            ui_widget_menu_refresh(key_event, global_state, global_state->w_menu_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_widget_menu_event_handler(key_event, global_state, global_state->w_menu_main, ui_musicbox_menu_item_action, STATE_MAIN_MENU, STATE_MUSICBOX_MENU);

        // 退出音乐盒模块（回主菜单）：释放文件列表（PSRAM）
        if (global_state->STATE == STATE_MAIN_MENU) {
            ui_musicbox_menu_on_exit();
        }

        break;


    /////////////////////////////////////////////
    // 音乐盒：播放中（解码 → 扬声器流式播放；D 暂停/恢复，←→ 音量，4/6 切曲，A 返回）
    /////////////////////////////////////////////

    case STATE_MUSICBOX_PLAYING:

        // 首次获得焦点：分配资源、打开解码器并启动播放
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_musicbox_playing_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_musicbox_playing_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 演化算法
    /////////////////////////////////////////////

    case STATE_GENETIC:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_app_genetic_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_app_genetic_refresh(key_event, global_state, 10);

        // 按键（实体键盘兜底）：A 返回主菜单；D 重新演化
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_MAIN_MENU;
        }
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
            ui_app_genetic_init(key_event, global_state);
        }
        // 触屏（全屏宫格软键退场后的显式入口，几何见 ui_genetic.h 的提示绘制）：
        // 页眉右侧 1/4 松手=返回主菜单；屏幕底部 24px 条带松手=重新演化。
        else if (key_event->touch_edge & TOUCH_EDGE_UP) {
            if (key_event->touch_down_y >= 0
                && key_event->touch_down_y < ui_std_header_height(global_state->ui_font)
                && key_event->touch_down_x >= UI_BACK_HOTSPOT_X0((int32_t)global_state->gfx->width)) {
                global_state->STATE = STATE_MAIN_MENU;
            }
            else if (key_event->touch_down_y >= (int32_t)global_state->gfx->height - 24) {
                ui_app_genetic_init(key_event, global_state);
            }
        }

        break;


    /////////////////////////////////////////////
    // 电子词典：前缀查询（候选菜单 + 固定软键盘；统一 GFX_FONT_ALPHA_12）
    /////////////////////////////////////////////

    case STATE_DICT_QUERY:

        // 注意：必须先更新 PREV_STATE 再调用事件处理——若处理中发生状态迁移（如退出回主菜单），
        // 依赖“PREV_STATE != STATE 才重绘”的目标状态才能触发首帧重绘
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_dict_query_event(key_event, global_state,
            STATE_MAIN_MENU, STATE_DICT_QUERY, STATE_DICT_DETAIL);

        break;

    /////////////////////////////////////////////
    // 电子词典：词条详情（字体 GFX_FONT_ALPHA_16）
    /////////////////////////////////////////////

    case STATE_DICT_DETAIL:

        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_dict_detail_event(key_event, global_state,
            STATE_DICT_QUERY, STATE_DICT_DETAIL);

        break;


    /////////////////////////////////////////////
    // 电源键锁屏：灭屏低功耗。进入/解锁由 main_event_handler 入口的电源键全局拦截处理；
    // 本分支仅负责 LED 心跳，其余事件一律忽略（锁屏期间 Core1 已停止事件生产，此处为兜底）
    /////////////////////////////////////////////

#if NANO_HAS_PWR_KEY
    case STATE_LOCK_SCREEN:

        global_state->PREV_STATE = global_state->STATE;

        // LED 心跳：每 3 秒短促闪烁 10ms，标识机器处于开机状态；颜色逐次循环（红黄绿青蓝紫白）。
        // 复用 misc_led_blink 异步机制（双核主循环的 misc_led_poll 推进熄灭）；
        // Core2 为 PMIC 自治 PWM 单色灯（颜色参数忽略），CoreS3 为 WS2812 灯带整带同色。
        if (global_state->timestamp >= s_lock_next_blink_ms) {
            s_lock_next_blink_ms = global_state->timestamp + LOCK_SCREEN_BLINK_PERIOD_MS;
            misc_led_blink(s_lock_blink_colors[s_lock_blink_color_idx],
                           LOCK_SCREEN_BLINK_BRIGHTNESS, LOCK_SCREEN_BLINK_ON_MS);
            s_lock_blink_color_idx = (s_lock_blink_color_idx + 1) % LOCK_SCREEN_BLINK_COLOR_NUM;
        }

        break;
#endif // NANO_HAS_PWR_KEY


    /////////////////////////////////////////////
    // 关机确认
    /////////////////////////////////////////////

    case STATE_SHUTDOWN:
    {
        // 首次获得焦点：复位滑块并整页重绘
        if (global_state->PREV_STATE != global_state->STATE) {
            s_shutdown_progress = 0;
            s_shutdown_dragging = 0;
            s_shutdown_grab_dx = 0;
            s_shutdown_drawn_progress = -1;
            s_shutdown_busy = 0;
            ui_shutdown_draw_page(key_event, global_state);
            gfx_refresh(global_state->gfx);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 关机进行中 / 失败提示中：忽略一切输入
        if (s_shutdown_busy) {
            break;
        }

        // ---- 实体键盘（触屏不派生键码，此处仅实体键；仍显式过滤软按键）----
        if (key_event->is_soft_key == 0) {
            // 长按 D 键：直接确认关机（保留原有硬键盘快捷方式）
            if (key_event->key_edge == -2 && key_event->key_code == NANO_KEY_enter) {
                ui_shutdown_do_shutdown(key_event, global_state);
                break;
            }
            // 短按/长按 A 键：取消关机，返回主菜单
            if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
                global_state->STATE = STATE_MAIN_MENU;
                break;
            }
        }

        // ---- 触屏：抓住推钮左右拖动 ----
        Shutdown_Ui u;
        ui_shutdown_layout(global_state, &u);

        // 页眉右侧「返回」热点（松手沿 + 按下点坐标，同文本控件约定）：触屏设备的取消入口
        if ((key_event->touch_edge & TOUCH_EDGE_UP)
            && key_event->touch_down_y >= 0
            && key_event->touch_down_y < ui_shutdown_header_h(global_state)
            && key_event->touch_down_x >= UI_BACK_HOTSPOT_X0((int32_t)global_state->gfx->width)) {
            global_state->STATE = STATE_MAIN_MENU;
            break;
        }

        // 按下沿：只锁存状态，不引发任何状态切换（命中判定只认推钮及其附近余量，
        // 点按轨道其它位置不会误触发；余量随轨道高度放大，小屏也易于抓取）
        if (key_event->touch_edge & TOUCH_EDGE_DOWN) {
            int32_t knob_cx = ui_shutdown_knob_cx(&u, s_shutdown_progress);
            int32_t cy = u.track_y + u.track_h / 2;
            int32_t hit_r = u.knob_r + UI_SHUTDOWN_GRAB_SLOP;
            int32_t dx = key_event->touch_down_x - knob_cx;
            int32_t dy = key_event->touch_down_y - cy;
            if (dx * dx + dy * dy <= hit_r * hit_r) {
                s_shutdown_dragging = 1;
                // 抓取点与钮心的偏差：拖动时保持相对位置，推钮不在手指下跳动
                s_shutdown_grab_dx = knob_cx - key_event->touch_down_x;
            }
        }

        // 拖动中：推钮跟随手指（逐帧读触点快照），越界由进度钳制消化
        if (s_shutdown_dragging && key_event->is_touching) {
            s_shutdown_progress = ui_shutdown_progress_from_x(&u, key_event->touch_x + s_shutdown_grab_dx);
        }

        // 松手沿：滑到位则关机，否则回弹到起点
        if (key_event->touch_edge & TOUCH_EDGE_UP) {
            if (s_shutdown_dragging) {
                s_shutdown_dragging = 0;
                if (s_shutdown_progress >= UI_SHUTDOWN_COMPLETE_PCT) {
                    ui_shutdown_do_shutdown(key_event, global_state);
                    break;
                }
                s_shutdown_progress = 0;
            }
        }

        // 仅在进度变化时重绘滑块（脏区推帧：拖动时只推轨道包围盒）
        if (s_shutdown_progress != s_shutdown_drawn_progress) {
            ui_shutdown_draw_slider(global_state, &u, s_shutdown_progress);
            s_shutdown_drawn_progress = s_shutdown_progress;
            gfx_refresh(global_state->gfx);
        }

        break;
    }


    /////////////////////////////////////////////
    // 设置：虚拟键盘输入数值
    /////////////////////////////////////////////

    case STATE_SETTING_INPUT:

        // 首次获得焦点：绘制编辑器。编辑值已由 ui_settings_editor_enter 捕获。
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_settings_editor_draw(global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_settings_editor_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // Animac终端：初始化
    /////////////////////////////////////////////

    case STATE_ANIMAC_INIT: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            // ANIMAC终端临时将文字编辑控件字体改为 GFX_FONT_ALPHA_12，退出时恢复
            s_animac_prev_ui_font = global_state->ui_font;
            global_state->ui_font = GFX_FONT_ALPHA_12;
            // 输入框：可编辑 input 控件（输入输出分离后为纯输入，不再承载历史与提示符）
            ui_widget_input_init(key_event, global_state, global_state->w_input_main, L"电子核桃控制台");
            // 终端模式：输入框动态高度（下沿锚定页脚上沿，随内容行数以1倍行高步进向上扩张）
            // + 关联只读日志区（页眉与输入框之间，几何随输入框联动，见 ui_widget_input_dyn_layout）
            global_state->w_input_main->dyn_height = 1;
            global_state->w_input_main->log_view = global_state->w_textarea_main;
            // 日志区：复用全局单例 textarea，嵌入（裸）模式——不绘制自己的页眉侧文本、不自行推帧
            ui_widget_textarea_init(key_event, global_state, global_state->w_textarea_main, UI_STR_BUF_MAX_LENGTH);
            global_state->w_textarea_main->is_bare = 1;
            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, UI_ANIMAC_STARTUP_MESSAGE, -1, 1);
            // 终端布局 + 首帧完整重绘（dyn 布局在绘制中按输入框内容行数自动重算，日志区同帧合成）
            ui_widget_input_dyn_layout(key_event, global_state, global_state->w_input_main);
            ui_draw_input_buffer(key_event, global_state, global_state->w_input_main);
        }
        global_state->PREV_STATE = global_state->STATE;

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    // NOTE 临时调试代码（后续删除）：初始化REPL前打印当前内存使用情况
    printf("[Animac] DRAM Free: %u | Largest: %u | PSRAM Free: %u | DMA Free: %u/%u\n",
        heap_caps_get_free_size(MALLOC_CAP_8BIT),
        heap_caps_get_largest_free_block(MALLOC_CAP_8BIT),
        heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
        heap_caps_get_free_size(MALLOC_CAP_DMA),
        heap_caps_get_largest_free_block(MALLOC_CAP_DMA));
#endif

        ui_animac_init(key_event, global_state);

        global_state->STATE = STATE_ANIMAC_CONSOLE;

        break;
    }


    /////////////////////////////////////////////
    // Animac终端：终端等待输入
    /////////////////////////////////////////////

    case STATE_ANIMAC_CONSOLE: {

        // 首次获得焦点：.load 指令载入的内容注入输入框（注入后内部已重绘）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_animac_apply_pending_input(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 编辑器模式：驱动解释器事件循环（定时器等异步任务），并将其输出行追加到日志区
        ui_animac_idle_pump(key_event, global_state);

        // 触屏序列属主路由（日志区/输入框两个手势机互斥）：DOWN 时按按下点归属，整个序列只喂给
        // 属主手势机——否则两侧 DOWN 判定的电平兜底（|| is_touching，取当前坐标）会在拖动穿越
        // 两区边界时把另一方中途激活，两个手势机同时滚动互抢（软键盘下输入框仅 1~3 行高，
        // 拖动极易穿越边界；曾表现为“软键盘下输入框偶尔无法滚动/滚动错乱”）
        // 属主 3（穿透）：DOWN 落在输入框且输入框无滚动余量（max_scroll==0，内容全可见）时，
        // 日志区与输入控件同喂——输入框手势机不滚动（无互抢），拖动穿透滚日志区、
        // 点按仍由输入控件定位光标（类聊天应用手感）。
        if (key_event->touch_edge & TOUCH_EDGE_DOWN) {
            Widget_Textarea_State *log_ta = global_state->w_textarea_main;
            Widget_Textarea_State *in_ta = &global_state->w_input_main->textarea;
            if (key_event->touch_down_x >= log_ta->x && key_event->touch_down_x < log_ta->x + log_ta->width
                && key_event->touch_down_y >= log_ta->y && key_event->touch_down_y < log_ta->y + log_ta->height) {
                s_animac_touch_owner = 1;
            }
            else {
                int32_t in_line_height = gfx_font_line_height(global_state->ui_font);
                int32_t in_scrollable = (in_ta->line_num * in_line_height > in_ta->height) ? 1 : 0;
                int32_t in_input_box =
                    (key_event->touch_down_x >= in_ta->x && key_event->touch_down_x < in_ta->x + in_ta->width
                     && key_event->touch_down_y >= in_ta->y && key_event->touch_down_y < in_ta->y + in_ta->height);
                s_animac_touch_owner = (in_input_box && !in_scrollable) ? 3 : 2;
            }
        }
        // 日志区触屏手势泵（滑动=像素滚动+惯性；手势机仅在序列起点位于日志区内激活，
        // 与输入框文本区不相交；点按不响应）。属主为输入框且触摸进行中时跳过（互斥）；
        // 非触摸帧照常喂入以推进惯性动画（新 DOWN 会先行终止旧惯性，见手势机实现）。
        // 滚动活动时就地重绘日志区（以 is_modified==0 包裹避免逐帧重排版，范式同
        // ui_widget_textarea_event_handler；bare 模式不动页眉）并推帧
        if (s_animac_touch_owner != 2 || !key_event->is_touching) {
            if (ui_widget_textarea_touch_handler(key_event, global_state, global_state->w_textarea_main) == 1) {
                global_state->w_textarea_main->is_modified = 0;
                ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
                global_state->w_textarea_main->is_modified = 1;
                gfx_refresh(global_state->gfx);
            }
        }

        // 注：软键盘切换请求消费与 Ctrl+0 绑定均已下沉到文本输入控件（ui.c），此处不再拦截

        // Ctrl+V（触屏软键盘）：恢复上次提交的输入内容到输入框
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->is_softkbd == 1
            && global_state->is_ctrl_enabled == 1
            && (key_event->key_code == NANO_KEY_v || key_event->key_code == NANO_KEY_V)) {
            global_state->is_ctrl_enabled = 0;
            ui_animac_restore_last_input(key_event, global_state);
            break;
        }

        // 退出语义统一封装：页眉“返回”按钮、16键键盘“退格”键、全键盘 Esc 键（后两者键码均为
        // NANO_KEY_esc）的【返回】语义一律经退出确认模态框（STATE_ANIMAC_EXIT_CONFIRM）+ 统一善后
        //（ui_app_animac_cleanup），语义完全一致。原语义保留：输入缓冲区非空时 Esc 仍是删除
        // 光标左侧字符（透传给控件），仅当缓冲区为空（且控件空闲态 state==0）时才是返回语义；
        // 输入法组字/选字/选符态（state 1/2/3）的 Esc 是取消组字语义，也透传给控件；
        // Ctrl+Esc 缓冲区非空时的强制退出为控件原语义（不经模态框，由下方兜底善后）
        // ① 页眉“返回”软按钮（UP 沿 + 按下点，与输入控件固有范式同）
        if ((key_event->touch_edge & TOUCH_EDGE_UP)
            && key_event->touch_down_y >= 0 && key_event->touch_down_y < ui_std_header_height(global_state->ui_font)
            && key_event->touch_down_x >= UI_BACK_HOTSPOT_X0((int32_t)global_state->gfx->width)) {
            global_state->STATE = STATE_ANIMAC_EXIT_CONFIRM;
            break;
        }
        // ② 16键“退格”键 / ③ 全键盘 Esc 键（NANO_KEY_esc）：仅当输入缓冲区为空时才拦截为返回语义
        if ((key_event->key_edge == -1 || key_event->key_edge == -2)
            && key_event->key_code == NANO_KEY_esc
            && global_state->w_input_main->state == 0
            && global_state->w_input_main->textarea.text[0] == L'\0') {
            global_state->STATE = STATE_ANIMAC_EXIT_CONFIRM;
            break;
        }

        // Animac控制台交互：Enter提交执行、Ctrl+Enter插入换行（通用输入框默认语义）。
        // 无Ctrl的Enter仍在此拦截而不走控件原生分支：控件原生提交路径会经
        // ui_widget_input_on_leave 收起软键盘/16键键盘，终端提交执行后键盘须保持展开。
        // 仅拦截控件空闲态（state==0）的Enter：组字/选字/选符态（state 1/2/3）的Enter
        // 是输入法分页/选定语义，透传给控件
        if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter
            && global_state->is_ctrl_enabled == 0 && global_state->w_input_main->state == 0) {
            global_state->w_input_main->state = 0;
            global_state->STATE = STATE_ANIMAC_RUNNING;
        }
        else {
            // 序列属主为日志区时对输入控件屏蔽触屏字段（保留按键/键盘事件），防电平兜底互抢
            Key_Event ke_input = *key_event;
            if (s_animac_touch_owner == 1 && (ke_input.is_touching || ke_input.touch_edge != 0)) {
                ke_input.is_touching = 0;
                ke_input.touch_edge = 0;
            }
            global_state->STATE = ui_widget_input_event_handler(&ke_input, global_state, global_state->w_input_main, STATE_MAIN_MENU, STATE_ANIMAC_CONSOLE, STATE_ANIMAC_RUNNING);
        }

        // 离开控制台：退出善后（兜底。正常退出统一经退出确认模态框“确认”路径，见上方
        // “退出语义统一封装”；此处防御控件其他返回路径漏网——模态框中转不算离开，
        // 故条件排除 STATE_ANIMAC_EXIT_CONFIRM）
        if (global_state->STATE != STATE_ANIMAC_CONSOLE && global_state->STATE != STATE_ANIMAC_RUNNING
            && global_state->STATE != STATE_ANIMAC_EXIT_CONFIRM) {
            ui_app_animac_cleanup(key_event, global_state);
        }

        break;
    }


    /////////////////////////////////////////////
    // Animac终端：执行中
    /////////////////////////////////////////////

    case STATE_ANIMAC_RUNNING: {

        global_state->PREV_STATE = global_state->STATE;

        Widget_Textarea_State *input_ta = &global_state->w_input_main->textarea;
        Widget_Textarea_State *log_ta = global_state->w_textarea_main;

        // 控制台输入缓冲区（一次性分配于PSRAM，避免占用任务栈与紧张的内部RAM；
        // 控制台单线程运行无重入；长度与控制台文本缓冲上限一致）
        static wchar_t *new_input = NULL;
        if (!new_input) new_input = (wchar_t*)platform_calloc(UI_STR_BUF_MAX_LENGTH, sizeof(wchar_t));
        wcsncpy(new_input, input_ta->text, UI_STR_BUF_MAX_LENGTH - 1);
        new_input[UI_STR_BUF_MAX_LENGTH - 1] = L'\0';

        // 记录本次提交的输入，供 Ctrl+V 恢复
        ui_animac_save_last_input(new_input);

        // 回显到日志区（"> "+输入），随后解释器输出累加于日志区尾部
        ui_animac_log_trim(global_state, (uint32_t)wcslen(new_input) + 4);
        ui_animac_console_append(log_ta->text, L"> ");
        ui_animac_console_append(log_ta->text, new_input);
        ui_animac_console_append(log_ta->text, L"\n");

        ui_animac_exec(key_event, global_state, new_input, log_ta->text);

        // 清空输入框、日志区强制滚底（智能滚动“首次更新必滚动”：提交轮的回显+输出无条件滚底）、
        // 整帧重绘（控制台模式同帧合成日志区）
        input_ta->text[0] = L'\0';
        input_ta->length = 0;
        input_ta->is_modified = 1;
        global_state->w_input_main->cursor_pos = -1;
        global_state->w_input_main->desired_x = -1;
        ui_animac_log_force_bottom(global_state);
        ui_draw_input_buffer(key_event, global_state, global_state->w_input_main);

        global_state->STATE = STATE_ANIMAC_CONSOLE;

        break;
    }


    /////////////////////////////////////////////
    // Animac终端：退出确认模态框（页眉“返回”拦截后的确认）
    /////////////////////////////////////////////

    case STATE_ANIMAC_EXIT_CONFIRM: {

        // 首次获得焦点：叠加绘制模态框（控制台画面保留在其下）。
        // 确认框是全局复用组件（ui.h），本状态只是它的一个宿主：弹出时 init 一次，
        // 之后每帧把触点喂给 _arm，松手沿用 _hit 结算。
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_confirm_modal_init(&s_animac_exit_modal, L"是否退出？", NULL, 0);
            ui_confirm_modal_draw(global_state, &s_animac_exit_modal);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 模态期间不泵编辑器事件循环（ui_animac_idle_pump 暂停）：
        // 避免异步输出触发整帧重绘覆盖模态框；运行时定时器随之暂停，对话框存续期间可接受

        // 按下反馈（见 ui.h 的 ui_confirm_modal_arm）：按住期间把那枚按钮画成
        // 「按下色」+ 标签下沉。判据是“现在松手会不会触发”（按下点与**当前点**都在
        // 同一枚按钮内）——拖出按钮立即恢复常态，拖回来又亮，从外面拖进来的不亮。
        if (!(key_event->touch_edge & TOUCH_EDGE_UP)
            && ui_confirm_modal_arm(global_state, &s_animac_exit_modal, key_event->is_touching,
                                    key_event->touch_down_x, key_event->touch_down_y,
                                    key_event->touch_x, key_event->touch_y)) {
            ui_confirm_modal_draw(global_state, &s_animac_exit_modal);   // 叠加式重绘（幂等）
        }

        // 触屏按钮（松手沿）：一次点按必须**按下→松开全在同一枚按钮内**
        // （按住后拖出去再松手 = 放弃，框保持打开；与 app/lib/button.scm 同一语义）。
        if (key_event->touch_edge & TOUCH_EDGE_UP) {
            int32_t hit = ui_confirm_modal_hit(global_state, &s_animac_exit_modal,
                                               key_event->touch_down_x, key_event->touch_down_y);
            int32_t rel = ui_confirm_modal_hit(global_state, &s_animac_exit_modal,
                                               key_event->touch_x, key_event->touch_y);
            if (hit != UI_CONFIRM_BTN_NONE && hit == rel) {
                if (hit == UI_CONFIRM_BTN_CONFIRM) {
                    // 确认退出：完整善后（解释器内存/字体/终端布局关联/日志区几何）后回主菜单
                    ui_app_animac_cleanup(key_event, global_state);
                    global_state->STATE = STATE_MAIN_MENU;
                }
                else {
                    // 留下：恢复控制台画面（整帧重绘，终端模式同帧合成日志区）并返回
                    ui_draw_input_buffer(key_event, global_state, global_state->w_input_main);
                    global_state->STATE = STATE_ANIMAC_CONSOLE;
                }
            }
            // 未构成一次完整点按：框留着，只把按下色恢复常态（否则会停在“按住”的样子）
            else if (ui_confirm_modal_arm(global_state, &s_animac_exit_modal, 0, 0, 0, 0, 0)) {
                ui_confirm_modal_draw(global_state, &s_animac_exit_modal);
            }
        }
        // A键（退格/返回）等价于“留下”
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_esc) {
            ui_draw_input_buffer(key_event, global_state, global_state->w_input_main);
            global_state->STATE = STATE_ANIMAC_CONSOLE;
        }

        break;
    }














    default:
        break;
    }

    return 0;
}



int32_t main_periodic_task(Key_Event *key_event, Global_State *global_state) {
    // 定期检查系统状态
    if (global_state->timer % 600 == 0) {
#ifdef ASR_ENABLED
        // ASR服务状态
        global_state->is_asr_server_up = check_asr_server_status();
#endif
#ifdef UPS_ENABLED
        // 锁屏期间跳过：read_ups_is_charging 内含 setLed 副作用（充电指示灯），
        // 会与锁屏 LED 心跳争用同一颗指示灯（无电源键平台 LOCK_SCREEN_ACTIVE 恒假，行为同原始）
        if (!LOCK_SCREEN_ACTIVE(global_state)) {
            global_state->ups_is_charging = read_ups_is_charging();
            global_state->ups_voltage = read_ups_voltage();
            global_state->ups_current = read_ups_current();
            global_state->ups_soc = read_ups_soc();
        }
#endif
    }
    // 逻辑时间戳
    global_state->timer = (global_state->timer == 2147483647) ? 0 : (global_state->timer + 1);

    // 自动关机：倒计时到期即优雅关机（关机失败则取消本次设置，避免反复触发）
    if (global_state->auto_shutdown_deadline != 0 &&
        global_state->timestamp >= global_state->auto_shutdown_deadline) {
        graceful_shutdown();
        global_state->auto_shutdown_deadline = 0;
        global_state->auto_shutdown_minutes = 0;
    }

    return 0;
}


int32_t main_deinit(Key_Event *key_event, Global_State *global_state) {
    // 鹦鹉笼（LLM）资源释放已提取至 ui_llm 模块
    ui_llm_deinit(global_state);

    gfx_close(global_state->gfx);

#ifdef ASR_ENABLED
    free(global_state->asr_output_buffer);
#endif

    free(global_state->w_textarea_main);

    free(global_state->w_input_main);

    free(global_state->w_menu_main);

#ifdef MATMUL_PTHREAD
    matmul_pthread_cleanup();
#endif

    return 0;
}