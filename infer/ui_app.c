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
#include "ui.h"
#include "ui_icon.h"
#include "ui_softkbd.h"
#include "ui_grid16kbd.h"
#include "ui_pinyin_ime.h"
#include "ui_linglong_ctrl.h"

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

#ifdef BADAPPLE_ENABLED
    #include "badapple.h"
#endif

#include "flip.h"

#include "ui_genetic.h"

#include "ui_tsp.h"

#include "ui_cloud.h"
#include "ui_llm.h"
#include "ephemeris.h"
#include "celestial.h"
#include "celestial_ocean.h"
#include "nongli.h"

#include "ui_color.h"
#include "ui_app.h"

#include "ui_animac.h"

#include "ui_spectrogram.h"

#include "ui_pedometer.h"

#include "ui_goldminer.h"

#include "ui_particlelife.h"

#include "ui_ripple.h"

#include "ui_scheme_app.h"

#include "ui_water.h"

#include "ui_tetris.h"

#include "ui_ebook.h"

#include "ui_ofdm.h"

#include "ui_musicbox.h"

#include "ui_dict.h"

#include "ui_calendar.h"

#define WALLPAPER_PATH (PLATFORM_ROOT_DIR "/wp.png")

// 全局变量（TODO 临时，后续要全部移到全局状态上下文中）

static uint64_t last_splash_timestamp = 0;

static int32_t s_album_count = 1;
static int32_t s_album_index = 0;
static char **s_album_path_list = NULL;
static int32_t s_album_is_autoplay = 0;
static uint64_t s_album_refresh_timestamp = 0;

// 指向图像缓冲区的指针
static uint8_t *s_image_file_buffer = NULL;
static size_t s_image_file_size = 0;
static char s_image_filename_buffer[128]; // 缓存图像文件名，用于确定是否要重新读取、重新解码
// 壁纸图像解码后的 RGB888 像素缓冲区（避免每次渲染都重新解码）
static uint8_t *s_image_rgb888_buffer = NULL;
static uint32_t s_image_width = 0;
static uint32_t s_image_height = 0;
static uint8_t s_image_decode_ready = 0;



// 玲珑仪主界面显式触屏控件是否接管触屏（1-接管；0-设置界面/其他状态）。
// get_input_event（输入生产侧）据此抑制全屏 4x4 宫格软键映射；显式控件只在
// 主天空界面使用，玲珑仪设置界面继续沿用十六宫格软键（遗留兜底），
// 实体硬键在两种界面下均照常工作。
static volatile int32_t s_linglong_touch_owned = 0;

static uint32_t s_animac_prev_ui_font = 0; // 进入 STATE_ANIMAC_* 之前的 ui_font，退出时恢复
// 控制台触屏序列属主：0=无序列 1=日志区 2=输入框 3=穿透（输入框无滚动余量：拖滚日志区+点按定位光标）
//（DOWN 时按按下点归属，整个序列只喂给属主手势机）
static int32_t s_animac_touch_owner = 0;






// ===============================================================================
// UI框架：获取输入事件（按键 + 触屏）
//
// 事件层只提供原始输入：实体按键、触屏→4x4宫格兼容映射（见下方注释）、
// 触屏电平/坐标（key_event->touch_*）。触屏手势的识别与语义解释归消费者
// （通用跟踪器 ui_swipe_tracker_*，见 ui.c），本层不参与。
// ===============================================================================

// ===============================================================================
// 触屏 → 4x4 宫格虚拟按键（兼容性适配层）
//
// 历史沿革：项目早期仅支持实体键盘，全部业务逻辑基于 NANO_KEY_* 键码事件；
// 后期主要设备只有触屏、无实体键盘。为复用既有键码处理逻辑，将触屏位置
// 按屏幕 4x4 宫格映射为虚拟键码（布局与实体十六键一致），与实体键互为备份。
// 该映射曾置于各平台按键HAL中（hal_key_m5esp/mp135/ncurses），旁路了干净的
// 触屏路径，造成架构混乱；现上移至输入事件层——HAL 只提供原始触屏（hal_touch）
// 与实体按键（hal_key），对触屏的一切解释（宫格映射、软键盘、滑动手势、
// 各业务控件的直读）统一在本层及上层完成。
//
// 软硬来源区分：宫格映射/触屏软键盘派生的键码与实体键盘键码取值相同、无法按
// 键码区分，故在事件上打标 Key_Event.is_soft_key（1-触屏派生软按键，0-实体键盘），
// 供实体键与触屏并存的平台（NANO_HAS_HW_KEYBOARD==1，如 Linux TTY）的消费者区分；
// 仅触屏设备（M5Core2/S3，NANO_HAS_HW_KEYBOARD==0）上该标记恒为 1。
// ===============================================================================
#define GRID16_X0 (0)
#define GRID16_X1 (SCREEN_WIDTH / 4 * 1)
#define GRID16_X2 (SCREEN_WIDTH / 4 * 2)
#define GRID16_X3 (SCREEN_WIDTH / 4 * 3)
#define GRID16_X4 (SCREEN_WIDTH)
#define GRID16_Y0 (0)
#define GRID16_Y1 (SCREEN_HEIGHT / 4 * 1)
#define GRID16_Y2 (SCREEN_HEIGHT / 4 * 2)
#define GRID16_Y3 (SCREEN_HEIGHT / 4 * 3)
#define GRID16_Y4 (SCREEN_HEIGHT)

// 菜单控件激活状态表：这些状态下菜单控件（ui_widget_menu_event_handler）直接消费
// 触屏流（拖动滚动/点按/顶栏退出），get_input_event 不再生成宫格软按键——否则同一次
// 触摸既产生触点流又产生软按键事件，ESP32 上两者经共享快照/事件队列两条通道到达渲染核，
// 时刻不同步，滞后的软按键事件会在菜单动作切换状态后泄漏给下一个状态造成误触发。
// （词典候选菜单 STATE_DICT_QUERY 不在此列：该状态软键盘常驻可见，宫格映射已被软键盘
//  路径禁用，且候选菜单依赖软键盘方向键导航。黄金矿工 STATE_GOLDMINER 非菜单控件，
//  但同样直接消费触屏流——返回虚拟按钮/点击放钩，故一并抑制。同理：电子书阅读
//  STATE_EBOOK_READING、本机自述 STATE_README、LLM 结果 STATE_LLM_AFTER_INFER 均经
//  文本控件手势机/自有拖动直接消费触屏流，抑制后拖动不再产生宫格软按键——否则长按
//  拖动的 -2 重复事件流会占满队列、触屏 DOWN/UP 可靠投递超时丢失，表现为拖动卡顿、
//  按键提示灯在拖动时被宫格 -1 事件点亮。OFDM 接收/环回等以宫格软按键为主要触屏
//  交互的状态不在此列。）
static int32_t ui_app_state_is_menu(int32_t state) {
    switch (state) {
        case STATE_MODEL_MENU:
        case STATE_GAME_MENU:
        case STATE_EBOOK:
        case STATE_OFDM_MENU:
        case STATE_MUSICBOX_MENU:
        case STATE_MUSICBOX_PLAYING: // 音乐盒播放页：显式触屏控件（播放/暂停/上下曲/音量/进度条/退出）
                                    // 直接消费触屏流，十六宫格软按键整体退场
                                    // （见 ui_musicbox.c 的 Musicbox_Ui / musicbox_handle_touch）
        case STATE_GOLDMINER: // 黄金矿工同样直接消费触屏流（返回虚拟按钮/点击放钩），抑制宫格软按键
        case STATE_TETRIS:   // 俄罗斯方块：虚拟按键/退出确认模态框直接消费触屏流，抑制宫格软按键
        case STATE_FLIP:     // 番茄表（电子沙漏）：顶栏/仪表/控制条全部为自绘触屏控件，
                             // 十六宫格软按键（旧版「节流/退出/画风/复位」角落格）已整体退场
                             // （见 ui_app_flip_event_handler 的触屏命中判定）
        case STATE_WATER:    // 水池：屏幕底部自绘控件栏+拖动旋转/拖球直接消费触屏流，抑制宫格软按键
        case STATE_EBOOK_READING:  // 电子书阅读：触屏拖动滚动/页脚按钮/返回热点直接消费触屏流
        case STATE_README:         // 本机自述：文本框拖动滚动+返回热点直接消费触屏流
        case STATE_LLM_AFTER_INFER: // LLM 结果：文本框拖动滚动+返回热点直接消费触屏流
        case STATE_ANIMAC_EXIT_CONFIRM: // 控制台退出确认模态框：触屏按钮直接消费触屏流（否则点按产生的宫格软按键会泄漏到下一状态误触发）
        case STATE_SHUTDOWN:        // 安全关机：手机式滑动关机滑块直接消费触屏流（抓推钮拖动/松手结算）。
                                    // 十六宫格软键整体退场：触屏不再能派生 enter 格直接关机；
                                    // 实体键盘的长按 D 关机 / A 取消照常保留（见 STATE_SHUTDOWN 分支）
        case STATE_SCHEME_APP_RUN:  // 灵机 App 运行态：触屏归 App（(native Touch) 直读），
                                    // **宫格软按键整体退场——包括右上角 esc 格**。
                                    // 这是 2026-10 的接口改造：A 键退出不再是宿主的统一交互，
                                    // 触屏不再能派生出“返回”；App 想退出就自己画按钮并调
                                    // (App.exit)（见 am_native_App.h）。
                                    // 实体键盘的 ESC 仍由宿主拦下，但只置“返回请求”位
                                    // （(App.back_requested)），不再直接销毁实例。
            return 1;
        default:
            return 0;
    }
}

// 文本输入控件宿主状态表：这些状态宿主 w_input_main（ui_widget_input_event_handler）。
// 触屏软按键的唯一来源是显式键盘（触屏软键盘 ui_softkbd / 16键虚拟键盘 ui_grid16kbd，
// 均在下方接管），全屏 4x4 宫格隐式映射在文本输入场景整体退场（见 ui_grid16kbd.h）。
// 必须抑制：否则文本区拖动滚动穿越宫格会产生 -2 重复事件流（1kHz 生产 vs Core0 每帧
// 仅消费 1 个键事件），挤占 8 深事件队列、触屏 DOWN/UP 可靠投递（1ms 超时）被饿死丢失，
// 表现为拖动滚动失效而点按幸免（点按的 DOWN/UP 在洪泛前到达）。
static int32_t ui_app_state_hosts_input_widget(int32_t state) {
    switch (state) {
        case STATE_LLM_INPUT:       // ui_llm.c
        case STATE_ANIMAC_CONSOLE:  // ui_app.c
        case STATE_OFDM_TX:         // ui_ofdm.c
        case STATE_OFDM_LOOP:       // ui_ofdm.c
            return 1;
        default:
            return 0;
    }
}

static uint8_t ui_app_map_touch_to_grid16_key(int32_t x, int32_t y) {
    if (y >= GRID16_Y0 && y < GRID16_Y1) {
        if (x >= GRID16_X0 && x <  GRID16_X1) return NANO_KEY_1;
        if (x >= GRID16_X1 && x <  GRID16_X2) return NANO_KEY_2;
        if (x >= GRID16_X2 && x <  GRID16_X3) return NANO_KEY_3;
        if (x >= GRID16_X3 && x <= GRID16_X4) return NANO_KEY_esc;
        else return NANO_KEY_IDLE;
    }
    else if (y >= GRID16_Y1 && y < GRID16_Y2) {
        if (x >= GRID16_X0 && x <  GRID16_X1) return NANO_KEY_4;
        if (x >= GRID16_X1 && x <  GRID16_X2) return NANO_KEY_5;
        if (x >= GRID16_X2 && x <  GRID16_X3) return NANO_KEY_6;
        if (x >= GRID16_X3 && x <= GRID16_X4) return NANO_KEY_shift;
        else return NANO_KEY_IDLE;
    }
    else if (y >= GRID16_Y2 && y < GRID16_Y3) {
        if (x >= GRID16_X0 && x <  GRID16_X1) return NANO_KEY_7;
        if (x >= GRID16_X1 && x <  GRID16_X2) return NANO_KEY_8;
        if (x >= GRID16_X2 && x <  GRID16_X3) return NANO_KEY_9;
        if (x >= GRID16_X3 && x <= GRID16_X4) return NANO_KEY_ctrl;
        else return NANO_KEY_IDLE;
    }
    else if (y >= GRID16_Y3 && y <= GRID16_Y4) {
        if (x >= GRID16_X0 && x <  GRID16_X1) return NANO_KEY_left;
        if (x >= GRID16_X1 && x <  GRID16_X2) return NANO_KEY_0;
        if (x >= GRID16_X2 && x <  GRID16_X3) return NANO_KEY_right;
        if (x >= GRID16_X3 && x <= GRID16_X4) return NANO_KEY_enter;
        else return NANO_KEY_IDLE;
    }
    else {
        return NANO_KEY_IDLE;
    }
}

// ===============================================================================
// 输入事件处理：背景、现状与 AI 必读原则
//
// 【历史背景】（项目维护者原话）
// “项目最开始只有硬按键，并设计了键码机制，几乎所有功能都是基于硬按键设计的。
//  后来项目开始支持触屏，为了平滑过渡，增加了将触屏事件映射为按键键码的机制，
//  也就是软按键机制，通过一个标记进行区分软按键和硬按键。目前正在对存量功能
//  进行触屏化改造，这就造成了触屏事件与软按键的混淆，不得不引入
//  ui_app_state_is_menu 作为临时修补方案。”
//
// 【现状：一次触摸的两条到达通道】
// 一次触屏点击会同时经两条通道到达消费者，且时刻不同步：
//   1) 触屏电平快照：key_event->touch_x / touch_y / is_touching，经跨核共享
//      内存传递，快，逐帧可见；
//   2) 宫格软按键事件：触屏按 4x4 宫格映射为 NANO_KEY_* 键码（或软键盘直接键码），
//      经 Core1→Core0 事件队列传递，慢，其下降沿可能在状态已切换后才到达，
//      从而“泄漏”到下一状态造成误触发（黄金矿工返回按钮曾踩此坑）。
// 软硬来源经 Key_Event.is_soft_key 打标区分：1-触屏派生软按键（宫格映射/软键盘），
// 0-实体键盘；仅触屏设备（M5Core2/S3，NANO_HAS_HW_KEYBOARD==0）上该标记恒为 1。
//
// 【按键/触屏事件处理原则】（触屏化改造过渡期，所有功能开发与修改必须遵守）
//   1. 存量功能默认仍按硬按键语义工作。仅应响应硬按键的功能，必须显式过滤
//      key_event->is_soft_key == 0（范式：ui_goldminer_event_handler），
//      否则同一次触摸会被“触屏流 + 软按键”重复响应。
//   2. 需要直接消费触屏的功能状态（虚拟按钮/点按/拖动等）：点按边沿认
//      key_event->touch_edge（TOUCH_EDGE_DOWN/UP，生产端高频检测+队列可靠投递，
//      亚帧点按不湮灭），按下点坐标取 touch_down_x/y（范式：ui_goldminer.c、
//      ui_calendar.c、ui.c 菜单控件）；拖动轨迹逐帧读 key_event->touch_x/touch_y
//      快照。一律不得跨层直读 hal_touch，也不再需要模块自持 prev 电平做沿检测。
//   3. 直接消费触屏流的状态，必须加入 ui_app_state_is_menu 抑制表，使本层不再
//      为该触摸生成宫格软按键，从根上杜绝队列通道的滞后事件泄漏到下一状态。
//   4. 会引发状态切换的触屏动作，必须在松手沿（touch_edge & TOUCH_EDGE_UP）
//      触发；按下沿只允许锁存坐标/状态。如此状态切换发生在手指抬起之后，
//      下一状态看到的触屏电平为 0，不会将本次触摸序列误当作新的点击
//      （范式：ui.c 菜单控件、ui_goldminer.c）。
//   5. ui_app_state_is_menu 是过渡期的临时修补：随着各功能逐个完成触屏化改造、
//      转为直接消费触屏流，该表随之扩充；存量功能全部完成改造后，软按键
//      兼容机制预期整体退场。
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
    uint8_t key_is_soft = 0; // 按键来源：0-实体键盘，1-触屏派生（宫格映射/软键盘）

    // 触屏统一采样（本函数在 Core1 每 1-2ms 轮询一次）：坐标与电平填入 key_event，
    // 供本轮所有消费者（宫格映射、软键盘、手势解释及各业务状态）使用，上层不再直接调 touch_read
    touch_read(&key_event->touch_x, &key_event->touch_y, &key_event->is_touching);

    // 触屏 UP 沿去抖（须在共享快照与边沿检测之前，保证三者语义一致）：松开认定要求连续
    // NANO_TOUCH_UP_DEBOUNCE 次轮询均无按压。背景：触屏挂在多设备共享的 I2C 总线上（Core2 的
    // FT6336 与 PMIC/IMU 同总线），拖动中偶发单次读抖动会产生假松开。若不去抖，假 UP+DOWN
    // 对会把一次拖动切成“点按（光标跳变）+重激活（拖动位移阈值重新累计）”循环，
    // 表现为触屏拖动失灵而点按定位正常（2026-08 定位）。抖动窗口内按按住对待
    //（key_event 为静态变量，未按压时 touch_read 不写坐标，自然保持上轮最后有效值）。
    if (!key_event->is_touching && s_input_touch_prev) {
        if (s_input_touch_up_pending + 1 < NANO_TOUCH_UP_DEBOUNCE) {
            s_input_touch_up_pending++;
            key_event->is_touching = 1;
        }
    }
    else {
        s_input_touch_up_pending = 0;
    }

    // 触屏电平共享快照：ESP32 上 Core0 渲染任务每帧取用本快照覆盖到其 key_event，
    // 高频电平样本不进入事件队列（见 linglong_m5core2.ino）
    global_state->touch_x = key_event->touch_x;
    global_state->touch_y = key_event->touch_y;
    global_state->is_touching = key_event->is_touching;

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

    // 触屏 → 4x4 宫格虚拟按键（兼容适配，见上方注释）：实体键优先，
    // 无实体键输入时按触点所在宫格映射为虚拟键码。
    // 菜单控件激活状态下抑制该映射（见 ui_app_state_is_menu 注释）：菜单直接消费
    // 触屏流，不再生成软按键事件，杜绝事件队列通道的滞后事件泄漏到下一状态。
    // 文本输入控件宿主状态同样整体抑制（见 ui_app_state_hosts_input_widget 注释）：
    // 软按键只来自显式键盘，且避免拖动滚动产生的软按键洪泛挤占触屏边沿事件。
    // 16键虚拟键盘（ui_grid16kbd）可见时亦抑制：触屏由键盘接管（见下）。
    if (key == NANO_KEY_IDLE && key_event->is_touching && !ui_app_state_is_menu(global_state->STATE)
        && !ui_app_state_hosts_input_widget(global_state->STATE)
        && !ui_grid16kbd_is_visible()
        && !ui_scheme_app_hosts_input_widget() // 灵机 App 的模态输入框显示期间同样抑制宫格
        && !(global_state->STATE == STATE_LINGLONG && s_linglong_touch_owned)) { // 玲珑仪主界面：显式触屏控件接管触屏，不再生成宫格软键
        key = ui_app_map_touch_to_grid16_key(key_event->touch_x, key_event->touch_y);
        key_is_soft = (key != NANO_KEY_IDLE); // 宫格映射命中的键来自触屏
    }
    uint8_t key_is_softkbd = 0;

    // 16键虚拟键盘（文本输入控件固有功能，文本输入场景下替代上方的全屏 4x4 宫格映射）：
    // 可见时，键盘区域内的触摸由键盘接管——命中按钮产生对应键码（走下方统一的边沿/长按
    // 机制，与旧宫格软按键等价）；键盘区域外的触屏不再产生任何宫格软按键。
    if (ui_grid16kbd_is_visible()) {
        uint8_t grid16_key = ui_grid16kbd_poll(key_event->touch_x, key_event->touch_y, key_event->is_touching);
        if (ui_grid16kbd_touch_claimed()) {
            key = grid16_key;
            key_is_soft = (grid16_key != NANO_KEY_IDLE); // 16键虚拟键盘键码来自触屏
        }
        else {
            key = NANO_KEY_IDLE; // 键盘区域外：不产生软按键
        }
    }

    // 触屏软键盘：可见时，键盘区域内的触摸由软键盘接管——吞掉触屏4x4网格键映射，
    // 改为注入软键盘键码（无边沿时为NANO_KEY_IDLE，仍走下方的边沿检测机制）。
    // 同时禁用键盘区域以外的十六键网格（仅保留右上角Esc键），防止误触。
    if (ui_softkbd_is_visible()) {
        uint8_t softkbd_key = ui_softkbd_poll(key_event->touch_x, key_event->touch_y, key_event->is_touching);
        if (ui_softkbd_touch_claimed()) {
            key = softkbd_key;
            key_is_soft = (softkbd_key != NANO_KEY_IDLE); // 软键盘键码来自触屏
            key_is_softkbd = (softkbd_key != NANO_KEY_IDLE);
        }
        else if (key != NANO_KEY_esc) {
            key = NANO_KEY_IDLE; // 软键盘激活时禁用十六键网格（Esc除外）
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
    global_state->ba_frame_count = 0;
    global_state->ba_begin_timestamp = 0;

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
        L"黄金矿工",
        L"Bad Apple!",
        L"康威生命游戏",
        L"遗传算法求解旅行商问题",
        L"遗传算法拟合图像",
        L"俄罗斯方块",
        L"粒子生命",
        L"水波",
        L"水池",
        L"日历",
        // L"体积云",  // 原入口：已移除（体积云渲染保留在天象仪“大气散射”档位中，见 celestial.c）
        // L"曼德博集" / L"生命游戏Scheme"：已并入“灵机应用程序”（业务资源目录 app/
        // 下的 Scheme App 动态列表，见 ui_scheme_app.c）
        // L"小鹦鹉笼",  // 原入口：小鹦鹉笼已并入“鹦鹉笼”的统一模型菜单（[轻] 前缀条目），注释保留备查
    };
    global_state->w_menu_main->title = L"小游戏";
    global_state->w_menu_main->items = game_menu_items;
    global_state->w_menu_main->item_num = 11;
    ui_widget_menu_init(key_event, global_state, global_state->w_menu_main);
}

int32_t game_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms) {
    switch (ms->current_item_index) {
        case 0: return STATE_SCHEME_APP_LIST;  // 灵机应用程序（Scheme App 动态列表）
        case 1: return STATE_GOLDMINER;
        case 2: return STATE_BADAPPLE;
        case 3: return STATE_GAMEOFLIFE;
        case 4: return STATE_GENETIC_TSP;
        case 5: return STATE_GENETIC;
        case 6: return STATE_TETRIS;
        case 7: return STATE_PARTICLELIFE;
        case 8: return STATE_RIPPLE;
        case 9: return STATE_WATER;
        case 10: return STATE_CALENDAR;
        // case 11: 原“体积云”入口已移除（保留 STATE_CLOUD 状态处理，当前无菜单入口）
        // 原“曼德博集”/“生命游戏Scheme”入口已并入“灵机应用程序”（case 0）
        // case 12: 原“小鹦鹉笼”入口已移除（并入鹦鹉笼模型菜单，条目带 [轻] 前缀）
        default: return STATE_MAIN_MENU;
    }
}











// ===============================================================================
// 主菜单
// ===============================================================================


static void ui_app_main_menu_grid16_refresh_button(
    Key_Event *key_event, Global_State *global_state,
    int32_t col, int32_t row, wchar_t *text,
    const char *icon_path,
    uint8_t cell_bg_R, uint8_t cell_bg_G, uint8_t cell_bg_B, uint8_t cell_bg_mode,
    uint8_t cell_text_R, uint8_t cell_text_G, uint8_t cell_text_B, uint8_t cell_text_mode
) {
    int32_t bx = (col == 0) ? 1 : 0;
    int32_t by = (row == 0) ? 1 : 0;
    // 网格布局：上下留白与顶/底栏高度一致（当前 UI 字体行高 + 1）
    const int32_t bar_height = 0; // gfx_font_line_height(global_state->ui_font) + 1;
    UI_Grid_Layout grid = ui_grid_layout_make(global_state->gfx->width, global_state->gfx->height, bar_height, bar_height, 0, 0, 4, 4);
    int32_t cell_cx = ui_grid_cell_center_x(&grid,col);
    int32_t cell_cy = ui_grid_cell_center_y(&grid,row);
    gfx_draw_rectangle(global_state->gfx, ui_grid_cell_x0(&grid,col)+bx, ui_grid_cell_y0(&grid,row)+by, ui_grid_cell_width(&grid)-1-bx, ui_grid_cell_height(&grid)-1-by, cell_bg_R, cell_bg_G, cell_bg_B, cell_bg_mode);
    // 图标（36x36，带PSRAM缓存）：中心点位于格子中心偏上8px
    ui_icon_draw_centered(global_state->gfx, icon_path, cell_cx, cell_cy - 8);
    // 文字：中心点位于图标下沿下方8px。后绘制文字，使其覆盖在图标上面
    gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, text, cell_cx, cell_cy - 8 + 18 + 8, cell_text_R, cell_text_G, cell_text_B, cell_text_mode);
}

void ui_widget_grid16_draw(Key_Event *key_event, Global_State *global_state) {
    wchar_t cell_text[4][4][2][10] = {
        { {L"[1]", L"番茄表",}, {L"[2]", L"鹦鹉笼",}, {L"[3]", L"玲珑仪",}, {L"[A]", L"返回",}, },
        { {L"[4]", L"电子书",}, {L"[5]", L"音乐盒",}, {L"[6]", L"时光集",}, {L"[B]", L"设置",}, },
        { {L"[7]", L"小游戏",}, {L"[8]", L"频谱仪",}, {L"[9]", L"寻呼机",}, {L"[C]", L"自述",}, },
        // { {L"[*]", L"计步器",}, {L"[0]", L"手电筒",}, {L"[#]", L"控制台",}, {L"[D]", L"关机",}, }, // 原入口（手电筒已征用为电子词典）
        { {L"[*]", L"计步器",}, {L"[0]", L"词典",}, {L"[#]", L"控制台",}, {L"[D]", L"关机",}, },
    };

    // 清屏
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_fill_white(global_state->gfx);
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_soft_clear(global_state->gfx);
    }

    uint8_t cell_bg_R = 0, cell_bg_G = 0, cell_bg_B = 0;
    uint8_t cell_text_R = 0, cell_text_G = 0, cell_text_B = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        cell_bg_R = 0xff;
        cell_bg_G = 0xff;
        cell_bg_B = 0xff;
        cell_text_R = 0x66;
        cell_text_G = 0x66;
        cell_text_B = 0x66;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        cell_bg_R = 30;
        cell_bg_G = 30;
        cell_bg_B = 32;
        cell_text_R = 0xcc;
        cell_text_G = 0xcc;
        cell_text_B = 0xcc;
    }

    // 所有格子的图标路径（暂定全部为 animac.png，后续逐格替换）。尺寸均为36x36。
    // 图标经 ui_icon_draw_centered 绘制：首次读取解码后常驻缓存于PSRAM，之后直接从缓存绘制
    const char *cell_icon_path[4][4] = {
        {PLATFORM_ROOT_DIR "/icon/fanqie.png", PLATFORM_ROOT_DIR "/icon/nano.png",
         PLATFORM_ROOT_DIR "/icon/linglong.png", PLATFORM_ROOT_DIR "/icon/home.png"},
        {PLATFORM_ROOT_DIR "/icon/ebook.png", PLATFORM_ROOT_DIR "/icon/music.png",
         PLATFORM_ROOT_DIR "/icon/album.png", PLATFORM_ROOT_DIR "/icon/settings.png"},
        {PLATFORM_ROOT_DIR "/icon/game.png", PLATFORM_ROOT_DIR "/icon/spectrogram.png",
         PLATFORM_ROOT_DIR "/icon/ptt.png", PLATFORM_ROOT_DIR "/icon/readme.png"},
        {PLATFORM_ROOT_DIR "/icon/pedometer.png", PLATFORM_ROOT_DIR "/icon/dict.png",
         PLATFORM_ROOT_DIR "/icon/animac.png", PLATFORM_ROOT_DIR "/icon/poweroff.png"},
    };

    for (int32_t row = 0; row < 4; row++) {
        for (int32_t col = 0; col < 4; col++) {
            ui_app_main_menu_grid16_refresh_button(key_event, global_state,
                col, row, cell_text[row][col][1],
                cell_icon_path[row][col],
                cell_bg_R, cell_bg_G, cell_bg_B, 1,
                cell_text_R, cell_text_G, cell_text_B, 1);

        }
    }

    // ui_draw_header(key_event, global_state, L"Nano-Pod", 1);
    // ui_draw_footer(key_event, global_state, L"(c) 2025-2026 BD4SUR", 1);
}

void ui_widget_grid16_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 番茄表
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_1) {
        global_state->STATE = STATE_FLIP;
    }
    // 鹦鹉笼
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_2) {
        init_model_menu(key_event, global_state);
        global_state->STATE = STATE_MODEL_MENU;
    }
    // 玲珑仪
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_3) {
        global_state->STATE = STATE_LINGLONG;
    }
    // 电子书
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_4) {
        ui_ebook_menu_init(key_event, global_state);
        global_state->STATE = STATE_EBOOK;
    }
    // 音乐盒
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_5) {
        ui_musicbox_menu_init(key_event, global_state);
        global_state->STATE = STATE_MUSICBOX_MENU;
    }
    // 时光集
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_6) {
        global_state->STATE = STATE_ALBUM;
    }
    // 小游戏
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_7) {
        init_game_menu(key_event, global_state);
        global_state->STATE = STATE_GAME_MENU;
    }
    // 频谱仪
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_8) {
        global_state->STATE = STATE_SPECTROGRAM;
    }
    // 寻呼机（OFDM 声波数传）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_9) {
        // TODO
        ui_ofdm_menu_init(key_event, global_state);
        global_state->STATE = STATE_OFDM_MENU;
    }
    // 电子词典（原“手电筒”入口及其颜色风格切换功能已取消，注释保留备查；
    // 颜色风格切换请使用“系统设置”中的正式按钮）
    // else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_0) {
    //     if (global_state->ui_color_style == UI_COLOR_LIGHT) {
    //         global_state->ui_color_style = UI_COLOR_DARK;
    //     }
    //     else {
    //         global_state->ui_color_style = UI_COLOR_LIGHT;
    //     }
    // }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_0) {
        // 索引就绪（缺失/失效则带进度构建）后进入查询状态；失败则停留在主菜单（错误画面已显示）
        if (ui_dict_enter(key_event, global_state) == 0) {
            global_state->STATE = STATE_DICT_QUERY;
        }
        else {
            sleep_in_ms(1500); // 错误画面停留片刻
            ui_widget_grid16_draw(key_event, global_state);
            gfx_refresh(global_state->gfx);
        }
    }
    // 返回
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_SPLASH_SCREEN;
    }
    // 设置
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_shift) {
        global_state->STATE = STATE_SETTING_MENU;
    }
    // 自述
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_ctrl) {
        global_state->STATE = STATE_README;
    }
    // 关机
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_enter) {
        global_state->STATE = STATE_SHUTDOWN;
    }
    // 计步器
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_left) {
        global_state->STATE = STATE_PEDOMETER;
    }
    // 控制台
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_right) {
        global_state->STATE = STATE_ANIMAC_INIT;
    }
    else {
        return;
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


// 启动画面底部两行（自下而上：农历 / 电量信息）的排版常量
#define SPLASH_FOOTER_LINE_GAP      (6)  // 两行之间的行间距（px）
#define SPLASH_FOOTER_BOTTOM_MARGIN (11) // 最底一行行框下边界距屏幕底边的留白（px，沿用旧电量行的留白）

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
    // 底部两行排版（按屏幕高度自适应）：最底一行是农历，其上一行是电量信息，
    // 最底一行行框下边界距屏幕底边 SPLASH_FOOTER_BOTTOM_MARGIN，
    // 两行之间留 SPLASH_FOOTER_LINE_GAP 行间距。
    // 居中绘制时行框下边界 = 行框中心 + (行高 - 行高/2)（见 gfx_font_draw_text_centered）。
    const int32_t nongli_line_h = gfx_font_line_height(GFX_FONT_ALPHA_12);
    const int32_t nongli_center_y = global_state->gfx->height - SPLASH_FOOTER_BOTTOM_MARGIN
                                  - (nongli_line_h - nongli_line_h / 2);

    const wchar_t *weekdays[] = {L"日", L"一", L"二", L"三", L"四", L"五", L"六"};
    // 日期时间行固定使用 12px alpha 平滑字体（窄屏也可完整显示）
    const uint32_t datetime_font = GFX_FONT_ALPHA_12;
    if (global_state->splash_clock_style == SPLASH_CLOCK_SMALL) {
        // 小样式：日期与 hh:mm:ss 合成一个字符串，整体居中
        swprintf(datetime_wcs_buffer, 33, L"%04d年%02d月%02d日 星期%ls %02d:%02d:%02d",
                 year, month, day, weekdays[timeinfo->tm_wday], hour, minute, second);
    }
    else {
        // 大样式：顶部只显示日期，时间由画面中央的七段码时钟表现
        swprintf(datetime_wcs_buffer, 33, L"%04d年%02d月%02d日 星期%ls", year, month, day, weekdays[timeinfo->tm_wday]);
    }
    // 背景（阴影）
    gfx_font_draw_text_centered(global_state->gfx, datetime_font, datetime_wcs_buffer, global_state->gfx->width / 2 + 1, 16 + 1, 0x66, 0x66, 0x66, 1);
    // 前景
    gfx_font_draw_text_centered(global_state->gfx, datetime_font, datetime_wcs_buffer, global_state->gfx->width / 2, 16, time_red, time_green, time_blue, 1);

    // 农历日期（最底一行，12px alpha 平滑字体）
    LunarDate *nongli = lunar_calculate(year, month, day, hour, minute, second, timezone);
    _mbstowcs(nongli_wcs_buffer, nongli->full_display, 33);
    // 背景（阴影）
    gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, nongli_wcs_buffer, global_state->gfx->width / 2 + 1, nongli_center_y + 1, 0x66, 0x66, 0x66, 1);
    // 前景
    gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, nongli_wcs_buffer, global_state->gfx->width / 2, nongli_center_y, nongli_red, nongli_green, nongli_blue, 1);

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
    // ui_app_linglong_draw_lite(key_event, global_state, (global_state->gfx->width - 128) / 2, 100,
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
// Bad Apple
// ===============================================================================

void ui_app_badapple_render_frame(Key_Event *key_event, Global_State *global_state) {
#ifdef BADAPPLE_ENABLED
    wchar_t ba_str[20];
    uint32_t center_x = global_state->gfx->width / 2;
    uint32_t center_y = global_state->gfx->height / 2;
    if (global_state->timestamp - global_state->ba_begin_timestamp >= 100 * global_state->ba_frame_count) {
        gfx_soft_clear(global_state->gfx);
        for (uint32_t row = 0; row < 64; row++) {
            uint32_t page_0 = bad_apple_10fps_64x64[global_state->ba_frame_count * 128 + row * 2];
            uint32_t page_1 = bad_apple_10fps_64x64[global_state->ba_frame_count * 128 + row * 2 + 1];
            for (uint32_t col = 0; col < 32; col++) {
                gfx_draw_point(global_state->gfx, col + (center_x - 32), row + (center_y - 32), 255, 255, 255, (page_0 >> (32 - 1 - col)) & 0x1);
            }
            for (uint32_t col = 32; col < 64; col++) {
                gfx_draw_point(global_state->gfx, col + (center_x - 32), row + (center_y - 32), 255, 255, 255, (page_1 >> (32 - 1 - (col-32))) & 0x1);
            }
        }

        swprintf(ba_str, 20, L"%04d / 2193", global_state->ba_frame_count);
        gfx_draw_textline_centered(global_state->gfx, ba_str, global_state->gfx->width/2, global_state->gfx->height - 8, 255, 255, 255, 1);

        gfx_refresh(global_state->gfx);
        global_state->ba_frame_count++;

        if (global_state->ba_frame_count > 2193) {
            global_state->ba_frame_count = 0;
            global_state->ba_begin_timestamp = global_state->timestamp;
        }
    }
#endif
}



// ===============================================================================
// Game of Life
// ===============================================================================

static uint8_t *s_ui_app_gol_field_0 = NULL;
static uint8_t *s_ui_app_gol_field_1 = NULL;
static uint8_t s_ui_app_gol_field_page = 0;
static uint64_t s_ui_app_gol_refresh_timestamp = 0;
static uint32_t s_ui_app_gol_step_count = 0;
static int32_t s_gol_width = 0;
static int32_t s_gol_height = 0;

// 每个格子对应的像素块边长（2px×2px 一个格子，计算量降为 1/4）
#define UI_APP_GOL_CELL_PX (2)

static inline uint8_t ui_app_gol_get_cell(uint8_t *field, int32_t w, int32_t h, int32_t x, int32_t y) {
    int32_t byte_index = (y * w + x) / 8;
    int32_t bit_rem = (y * w + x) % 8;
    return ((field[byte_index] & ((uint8_t)0x80u >> bit_rem)) != 0);
}

static inline void ui_app_gol_set_cell(uint8_t *field, int32_t w, int32_t h, int32_t x, int32_t y, uint8_t value) {
    int32_t byte_index = (y * w + x) / 8;
    int32_t bit_rem = (y * w + x) % 8;
    uint8_t oldv = field[byte_index];
    field[byte_index] = (oldv & ~((uint8_t)(0x80u >> bit_rem))) | ((uint8_t)(!!value << (7 - bit_rem)));
}

void ui_app_gol_init(Key_Event *key_event, Global_State *global_state, int32_t gol_width, int32_t gol_height) {
    s_gol_width = gol_width;
    s_gol_height = gol_height;

    s_ui_app_gol_step_count = 0;
    s_ui_app_gol_field_page = 0;
    s_ui_app_gol_refresh_timestamp = global_state->timestamp;
    uint64_t ts = global_state->timestamp;

    // 重新进入/刷新时先释放旧场，避免覆盖式分配造成泄漏（初次为NULL，free(NULL)安全）
    if (s_ui_app_gol_field_0) free(s_ui_app_gol_field_0);
    if (s_ui_app_gol_field_1) free(s_ui_app_gol_field_1);
    s_ui_app_gol_field_0 = (uint8_t*)platform_calloc(gol_width * gol_height / 8, sizeof(uint8_t));
    s_ui_app_gol_field_1 = (uint8_t*)platform_calloc(gol_width * gol_height / 8, sizeof(uint8_t));

    for (uint32_t x = 0; x < gol_width; x++) {
        for (uint32_t y = 0; y < gol_height; y++) {
            uint8_t s = random_u32(&ts) % 2;
            ui_app_gol_set_cell(s_ui_app_gol_field_0, gol_width, gol_height, x, y, s);
            ui_app_gol_set_cell(s_ui_app_gol_field_1, gol_width, gol_height, x, y, s);
        }
    }
}

void ui_app_gol_render_frame(Key_Event *key_event, Global_State *global_state) {
    // 节流：不大于50fps
    // if (global_state->timestamp - s_ui_app_gol_refresh_timestamp < 20) {
    //     return;
    // }
    // s_ui_app_gol_refresh_timestamp = global_state->timestamp;
    gfx_soft_clear(global_state->gfx);

    uint32_t total_count = 0;

    uint8_t *field     = (s_ui_app_gol_field_page) ? s_ui_app_gol_field_0 : s_ui_app_gol_field_1;
    uint8_t *field_new = (s_ui_app_gol_field_page) ? s_ui_app_gol_field_1 : s_ui_app_gol_field_0;
    for (uint32_t x = 0; x < s_gol_width; x++) {
        for (uint32_t y = 0; y < s_gol_height; y++) {
            // 获取某个格子的8邻域
            uint32_t count = 0;
            uint32_t x_a = (x == 0) ? (s_gol_width-1) : (x-1);
            uint32_t x_b = (x == (s_gol_width-1)) ? 0 : (x+1);
            uint32_t y_a = (y == 0) ? (s_gol_height-1) : (y-1);
            uint32_t y_b = (y == (s_gol_height-1)) ? 0 : (y+1);
            uint8_t n1 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height, x_a, y_a); count += (n1 != 0);
            uint8_t n2 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height,  x , y_a); count += (n2 != 0);
            uint8_t n3 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height, x_b, y_a); count += (n3 != 0);
            uint8_t n4 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height, x_a,  y ); count += (n4 != 0);
            uint8_t n5 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height,  x ,  y ); // self
            uint8_t n6 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height, x_b,  y ); count += (n6 != 0);
            uint8_t n7 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height, x_a, y_b); count += (n7 != 0);
            uint8_t n8 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height,  x , y_b); count += (n8 != 0);
            uint8_t n9 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height, x_b, y_b); count += (n9 != 0);

            uint8_t new_state = 0;
            if (n5 == 0) {
                new_state = (count == 3) ? 1 : 0;
            }
            else {
                new_state = (count == 2 || count == 3) ? 1 : 0;
            }

            ui_app_gol_set_cell(field_new, s_gol_width, s_gol_height, x, y, new_state);

            if (new_state) {
                // 每格绘制为 2×2 像素块
                gfx_draw_rectangle(global_state->gfx, (uint32_t)(x * UI_APP_GOL_CELL_PX), (uint32_t)(y * UI_APP_GOL_CELL_PX),
                    UI_APP_GOL_CELL_PX, UI_APP_GOL_CELL_PX, 0, 255, 255, 1);
                total_count++;
            }
        }
    }

    wchar_t text[100];
    swprintf(text, 100, L"康威生命游戏 | 迭代:%u | 存活:%u | 密度:%.2f%%", s_ui_app_gol_step_count, total_count, (float)total_count / (float)(s_gol_width * s_gol_height) * 100);
    gfx_draw_rectangle(global_state->gfx, 0, 0, global_state->gfx->width, 12, 39, 39, 39, 3);
    gfx_draw_textline(global_state->gfx, text, 0, 0, 255, 255, 255, 1);

    gfx_refresh(global_state->gfx);
    s_ui_app_gol_field_page = 1 - s_ui_app_gol_field_page;
    s_ui_app_gol_step_count++;
}


// ===============================================================================
// 番茄表（电子沙漏 / FLIP 流体）
// ===============================================================================
//
// 2026-11 GUI 重做（触屏化）：
//   · 本状态已列入 ui_app_state_is_menu 抑制表：输入层不再为本状态的触摸生成任何
//     十六宫格软按键（旧版的「节流 / 退出 / 画风 / 复位 / 调整节流度」角落提示已整体
//     退场）。所有交互都改为屏幕自绘控件；实体键盘（is_soft_key==0）保留一份兜底：
//     ESC 退出、←/→ 画风、1 节流、C 节流度、D 复位。
//   · 布局（320x240 设计基准，按画布宽高等比缩放）：
//       顶栏 30px   状态灯 + 标题「电子沙漏」+ 状态胶囊 + 模式胶囊 + 数码计时 + 「退出」
//       中央舞台    4:3 流体视口（与屏幕同宽高比 ⇒ 物理初始化参数 k 不变）+ 深色边框
//       左仪表      进度竖槽 + 百分比 + 余粒数（自由模式：粒子数 + 矩形槽）
//       右仪表      节流度竖槽（可拖动/点按）+ 百分比 + 当前值（自由模式：模式信息）
//       底部控制条  复位 / 画风（粒子·网格·叠加）/ 节流开关 / 配色 / 流量读数
//   · 自由流体模式：顶栏「自由/沙漏」胶囊切换（键盘 2 兜底）。自由模式用同一 FLIP
//     求解器，只把固体边界换成矩形外壁（flip_init 的 hourglass_enabled=0）、粒子初始
//     沉降在槽底，并**关闭全部节流机制**；没有流完判定/报警，计时退化为秒表。
//   · 色彩风格：底部「配色」胶囊打开模态选择器，5 档——黄（暖沙，默认）/ 蓝 / 绿 /
//     青 / 彩（原彩色：蓝粒子 + 彩虹密度网格）。风格同时驱动流体颜色（flip.c 的
//     flip_set_color_style，纯渲染）与 UI 强调色（仪表/胶囊/按钮/完成态描边）。
//   · 沙漏模式的物理与计时逻辑一行未改：dt、重力源与换算、节流自适应公式、翻转判定、
//     报警状态机、粒子流量统计全部沿用；render_flip 只是把流体画进舞台视口（center/
//     width/height 参数），视口与屏幕同宽高比，因此网格分辨率、沙漏几何、粒子数不变。
// ===============================================================================

// ---- 触屏控件标识 ----------------------------------------------------------
#define FLIP_UI_NONE       (-1)
#define FLIP_UI_EXIT       (0)
#define FLIP_UI_RESET      (1)
#define FLIP_UI_VIEW       (2)
#define FLIP_UI_THROTTLE   (3)
#define FLIP_UI_SLIDER     (4)
#define FLIP_UI_PALETTE    (5)
#define FLIP_UI_MODE       (6)

// ---- 色彩风格（与 flip.c 的 flip_set_color_style 一一对应）------------------
#define FLIP_COLOR_STYLES  (5)   // 0黄 1蓝 2绿 3青 4彩（原彩色）
static int32_t s_ui_flip_color_style = 0;
static int32_t s_ui_flip_picker_open = 0;
static int32_t s_ui_flip_picker_pressed = FLIP_UI_NONE;

// 自由流体模式：0=沙漏（沙漏实体 + 节流），1=自由流体（矩形舞台、无沙漏壁、无节流）。
// 切换时重新初始化流体（两种模式的固体边界不同），计时器重新开始。
static int32_t s_ui_flip_free_mode = 0;

// ---- 画风（与 s_ui_flip_setting_count 同一套取值）--------------------------
#define FLIP_VIEW_PARTICLE (0)   // 只看粒子
#define FLIP_VIEW_GRID     (1)   // 只看网格（密度着色）
#define FLIP_VIEW_BOTH     (2)   // 粒子 + 网格叠加

static int32_t s_ui_flip_setting_count = FLIP_VIEW_PARTICLE;
static int32_t s_ui_flip_show_particles = 1;
static int32_t s_ui_flip_show_grid = 0;
static int32_t s_ui_flip_is_throttle = 0;
static int32_t s_ui_flip_throttle = 0;
static int32_t s_ui_flip_init_throttle = 50;

static int32_t s_ui_flip_last_upper_count = -1; // 用于计算粒子流量（-1 = 尚未建立基线）
static uint64_t s_ui_flip_last_upper_count_timestamp = 0; // 用于计算粒子流量
static uint64_t s_ui_fanqie_start_timestamp = 0;
static uint64_t s_ui_fanqie_stop_timestamp = 0;
static int32_t s_ui_fanqie_is_running = 0;

static uint64_t s_ui_fanqie_alarm_count = 0;
static int32_t  s_ui_fanqie_alarm_phase = 0;      // 非阻塞报警状态机相位：0=空闲 1=鸣震段 2=静默段
static uint64_t s_ui_fanqie_alarm_phase_ts = 0;   // 当前相位起始时间戳

// 触屏控件交互状态（按下点锁存 + 松手沿结算，项目触屏范式）
static int32_t s_ui_flip_ui_pressed = FLIP_UI_NONE;
static int32_t s_ui_flip_ui_pressed_view = FLIP_VIEW_PARTICLE;
static int32_t s_ui_flip_ui_slider_active = 0;

// ---- 配色 ------------------------------------------------------------------
// 深/浅两套主题跟随全局 ui_color_style；沙漏舞台本身恒为深色（流体是自发光画面，
// 浅色主题下就是一块嵌进面板的深色窗口），只切换外部 chrome 的配色。
// 强调色（accent / accent_dim / accent_off / on_accent）不写死，由当前色彩风格派生：
// 深色主题取明亮色、浅色主题取饱和深色，保证两种主题下的对比度。
typedef struct {
    uint8_t bg_top[3];
    uint8_t bg_bot[3];
    uint8_t card_edge[3];     // 舞台外框
    uint8_t panel[3];         // 舞台面板 / 沙漏实体（同色 ⇒ 圆角接缝不可见）
    uint8_t surface[3];       // 控件底色
    uint8_t surface_press[3];
    uint8_t surface_edge[3];
    uint8_t accent[3];        // 以下四项由色彩风格派生（见 flip_ui_palette）
    uint8_t accent_dim[3];
    uint8_t accent_off[3];    // 关闭态下的节流仪表（弱化但仍可读）
    uint8_t on_accent[3];
    uint8_t text[3];
    uint8_t dim[3];
    uint8_t timer[3];
    uint8_t track[3];
    uint8_t wall_edge[3];
    uint8_t fluid_top[3];
    uint8_t fluid_bot[3];
    uint8_t separator_alpha;
} Flip_Ui_Palette;

static const uint8_t flip_accent_dark[FLIP_COLOR_STYLES][3] = {
    {255, 181, 71},   // 黄（暖沙）
    {86, 156, 255},   // 蓝
    {86, 214, 132},   // 绿
    {86, 214, 214},   // 青
    {150, 172, 196},  // 彩（流体为原彩色，UI 用低饱和浅蓝灰强调）
};
static const uint8_t flip_accent_light[FLIP_COLOR_STYLES][3] = {
    {232, 138, 31},
    {37, 99, 235},
    {22, 163, 74},
    {8, 145, 178},
    {100, 116, 139},  // 彩：浅蓝灰在亮底上太弱，改用同色相的低饱和中灰蓝
};
static const wchar_t *const flip_style_names[FLIP_COLOR_STYLES] = {
    L"黄", L"蓝", L"绿", L"青", L"彩"
};

static const uint8_t *flip_style_accent(int32_t style, int32_t is_light) {
    if (style < 0 || style >= FLIP_COLOR_STYLES) style = 0;
    return is_light ? flip_accent_light[style] : flip_accent_dark[style];
}

// 强调色上的文字色：按亮度选深/浅，亮底配深字、暗底配白字
static void flip_style_on_accent(const uint8_t *a, uint8_t out[3]) {
    int32_t lum = ((int32_t)a[0] * 299 + (int32_t)a[1] * 587 + (int32_t)a[2] * 114) / 1000;
    if (lum > 140) { out[0] = 28; out[1] = 19; out[2] = 5; }
    else { out[0] = 255; out[1] = 255; out[2] = 255; }
}

static void flip_ui_palette(Global_State *global_state, Flip_Ui_Palette *p) {
    static const Flip_Ui_Palette dark = {
        .bg_top        = {8, 11, 18},
        .bg_bot        = {17, 23, 35},
        .card_edge     = {60, 74, 100},
        .panel         = {25, 32, 46},
        .surface       = {28, 36, 50},
        .surface_press = {48, 62, 84},
        .surface_edge  = {54, 68, 92},
        .text          = {236, 242, 250},
        .dim           = {138, 152, 172},
        .timer         = {250, 246, 238},
        .track         = {38, 48, 64},
        .wall_edge     = {84, 106, 142},
        .fluid_top     = {8, 12, 19},
        .fluid_bot     = {3, 5, 9},
        .separator_alpha = 70,
    };
    static const Flip_Ui_Palette light = {
        .bg_top        = {247, 249, 253},
        .bg_bot        = {228, 235, 245},
        .card_edge     = {196, 207, 224},
        .panel         = {25, 32, 46},
        .surface       = {255, 255, 255},
        .surface_press = {224, 234, 247},
        .surface_edge  = {206, 217, 232},
        .text          = {28, 36, 50},
        .dim           = {106, 121, 140},
        .timer         = {28, 36, 50},
        .track         = {219, 227, 239},
        .wall_edge     = {84, 106, 142},
        .fluid_top     = {8, 12, 19},
        .fluid_bot     = {3, 5, 9},
        .separator_alpha = 36,
    };
    int32_t is_light = (global_state->ui_color_style == UI_COLOR_LIGHT);
    *p = is_light ? light : dark;

    const uint8_t *a = flip_style_accent(s_ui_flip_color_style, is_light);
    const uint8_t *mixc = is_light ? p->surface : p->panel;
    int32_t dim_k = is_light ? 82 : 68;   // accent_dim = accent * dim_k%
    int32_t off_w = is_light ? 52 : 38;   // accent_off = accent:底色 = off_w : (100-off_w)
    for (int32_t i = 0; i < 3; i++) {
        p->accent[i] = a[i];
        p->accent_dim[i] = (uint8_t)((int32_t)a[i] * dim_k / 100);
        p->accent_off[i] = (uint8_t)(((int32_t)a[i] * off_w
                                     + (int32_t)mixc[i] * (100 - off_w)) / 100);
    }
    flip_style_on_accent(a, p->on_accent);
}

// ---- 布局几何（绘制与命中判定共用同一份）-----------------------------------
typedef struct {
    int32_t w;
    int32_t h;
    int32_t header_h;
    int32_t title_x, title_y;
    int32_t chip_x, chip_y, chip_w, chip_h;
    int32_t mode_x, mode_y, mode_w, mode_h, mode_r;
    int32_t timer_y, timer_right;
    int32_t seg_len, seg_thick, seg_gap;
    int32_t exit_x, exit_y, exit_w, exit_h, exit_r;
    // 舞台
    int32_t card_x, card_y, card_w, card_h, card_r;
    int32_t fluid_x, fluid_y, fluid_w, fluid_h;
    // 左右仪表
    int32_t rail_y, rail_h;
    int32_t lrail_x, lrail_w, lrail_cx;
    int32_t rrail_x, rrail_w, rrail_cx;
    int32_t label_cy;
    int32_t gauge_w, gauge_h, gauge_y;
    int32_t lgauge_x, rgauge_x;
    int32_t gauge_value_cy;
    int32_t gauge_caption_cy;
    int32_t knob_r;
    // 底部控制条
    int32_t dock_y, dock_h;
    int32_t btn_y, btn_h, btn_r;
    int32_t reset_x, reset_w;
    int32_t view_x, view_w, view_cell_w;
    int32_t thr_x, thr_w;
    int32_t pal_x, pal_w;
    int32_t stat_cx, stat_y1, stat_y2;
    // 配色选择弹层（模态）
    int32_t pick_x, pick_y, pick_w, pick_h, pick_r;
    int32_t pick_title_x, pick_title_y;
    int32_t sw_x, sw_y, sw_w, sw_h, sw_gap, sw_r;
} Flip_Ui;

// ---- 布局 ------------------------------------------------------------------
static void flip_ui_layout(Nano_GFX *gfx, Flip_Ui *u) {
    memset(u, 0, sizeof(*u));
    u->w = (int32_t)gfx->width;
    u->h = (int32_t)gfx->height;

    float sx = (float)u->w / 320.0f;
    float sy = (float)u->h / 240.0f;
    float rs = (sx < sy) ? sx : sy;
#define FX(v) ((int32_t)lroundf((float)(v) * sx))
#define FY(v) ((int32_t)lroundf((float)(v) * sy))
#define FR(v) ((int32_t)lroundf((float)(v) * rs))

    // 顶栏
    u->header_h = FY(30);
    u->title_x = FX(8);
    u->title_y = FY(5);
    u->chip_h = FY(17);
    u->chip_x = FX(78);      // 标题「电子沙漏」= 4 个 16px 全角字（8..72）
    u->chip_y = FY(6);
    u->chip_w = FX(54);
    u->mode_w = FX(33);
    u->mode_h = FY(17);
    u->mode_x = u->chip_x + u->chip_w + FX(2);
    u->mode_y = u->chip_y;
    u->mode_r = u->mode_h / 2;
    u->timer_y = FY(4);
    u->seg_len = FR(8);
    if (u->seg_len < 4) u->seg_len = 4;
    u->seg_thick = FR(2);
    if (u->seg_thick < 1) u->seg_thick = 1;
    u->seg_gap = FR(2);
    if (u->seg_gap < 1) u->seg_gap = 1;
    u->exit_w = FX(46);
    u->exit_h = FY(20);
    u->exit_r = u->exit_h / 2;
    u->exit_x = u->w - FX(8) - u->exit_w;
    u->exit_y = FY(5);
    u->timer_right = u->exit_x - FX(8);

    // 中央舞台：流体视口 224x168（4:3），四周 4px 面板边框
    u->fluid_w = FX(224);
    u->fluid_h = FY(168);
    u->card_r = FR(10);
    u->card_w = u->fluid_w + 2 * FX(4);
    u->card_h = u->fluid_h + 2 * FY(4);
    u->card_x = (u->w - u->card_w) / 2;
    u->card_y = u->header_h + FY(2);
    u->fluid_x = u->card_x + FX(4);
    u->fluid_y = u->card_y + FY(4);

    // 左右仪表
    u->rail_y = u->card_y;
    u->rail_h = u->card_h;
    u->lrail_x = 0;
    u->lrail_w = u->card_x;
    u->lrail_cx = u->lrail_x + u->lrail_w / 2;
    u->rrail_x = u->card_x + u->card_w;
    u->rrail_w = u->w - u->rrail_x;
    u->rrail_cx = u->rrail_x + u->rrail_w / 2;
    u->label_cy = u->rail_y + FY(10);
    u->gauge_w = FX(10);
    if (u->gauge_w < 6) u->gauge_w = 6;
    u->gauge_h = FY(88);
    u->gauge_y = u->rail_y + FY(22);
    u->lgauge_x = u->lrail_cx - u->gauge_w / 2;
    u->rgauge_x = u->rrail_cx - u->gauge_w / 2;
    u->gauge_value_cy = u->rail_y + FY(118);
    u->gauge_caption_cy = u->rail_y + FY(138);
    u->knob_r = FR(5);
    if (u->knob_r < 3) u->knob_r = 3;

    // 底部控制条
    u->dock_y = u->card_y + u->card_h;
    u->dock_h = u->h - u->dock_y;
    u->btn_h = FY(26);
    if (u->btn_h > u->dock_h - FY(4)) u->btn_h = u->dock_h - FY(4);
    if (u->btn_h < 12) u->btn_h = 12;
    u->btn_y = u->dock_y + (u->dock_h - u->btn_h) / 2;
    u->btn_r = FR(8);
    u->reset_x = FX(6);
    u->reset_w = FX(50);
    u->view_x = FX(60);
    u->view_w = FX(90);
    u->view_cell_w = u->view_w / 3;
    u->thr_x = FX(154);
    u->thr_w = FX(50);
    u->pal_x = FX(208);
    u->pal_w = FX(46);
    u->stat_cx = FX(286);
    u->stat_y1 = u->dock_y + FY(9);
    u->stat_y2 = u->dock_y + FY(23);

    // 配色选择弹层：屏幕居中的圆角卡片（标题 + 5 个大色块）
    u->pick_w = FX(300);
    u->pick_h = FY(66);
    u->pick_x = (u->w - u->pick_w) / 2;
    u->pick_y = (u->h - u->pick_h) / 2;
    u->pick_r = FR(10);
    u->pick_title_x = u->pick_x + FX(10);
    u->pick_title_y = u->pick_y + FY(6);
    u->sw_gap = FX(4);
    u->sw_w = (u->pick_w - FX(18) - 4 * u->sw_gap) / FLIP_COLOR_STYLES;
    u->sw_h = FY(34);
    u->sw_r = FR(8);
    u->sw_x = u->pick_x + (u->pick_w - (FLIP_COLOR_STYLES * u->sw_w
                                        + (FLIP_COLOR_STYLES - 1) * u->sw_gap)) / 2;
    u->sw_y = u->pick_y + FY(26);

#undef FX
#undef FY
#undef FR
}

// ---- 绘图小工具 ------------------------------------------------------------
static void flip_ui_round(Nano_GFX *g, int32_t x, int32_t y, int32_t w, int32_t h, int32_t r,
                          const uint8_t c[3], uint8_t mode) {
    if (w <= 0 || h <= 0) return;
    int32_t rm = (w < h) ? w : h;
    rm /= 2;
    if (r < 0) r = 0;
    if (r > rm) r = rm;
    gfx_draw_rounded_rectangle(g, x, y, w, h, r, r, r, r, c[0], c[1], c[2], mode);
}

static void flip_ui_circle(Nano_GFX *g, int32_t cx, int32_t cy, int32_t r,
                           const uint8_t c[3], uint8_t mode) {
    if (r <= 0) return;
    flip_ui_round(g, cx - r, cy - r, r * 2, r * 2, r, c, mode);
}

static void flip_ui_text(Nano_GFX *g, uint32_t font, const wchar_t *s,
                         int32_t x, int32_t y_top, const uint8_t c[3], uint8_t mode) {
    gfx_font_draw_text(g, font, (wchar_t *)s, x, y_top, c[0], c[1], c[2], mode);
}

static void flip_ui_text_centered(Nano_GFX *g, uint32_t font, const wchar_t *s,
                                  int32_t cx, int32_t cy, const uint8_t c[3], uint8_t mode) {
    gfx_font_draw_text_centered(g, font, (wchar_t *)s, cx, cy, c[0], c[1], c[2], mode);
}

// 垂直渐变背景（8px 条带，条带按全局 y 对齐：与局部重绘语义一致）
static void flip_ui_bg(Nano_GFX *g, const Flip_Ui_Palette *p) {
    int32_t H = (int32_t)g->height;
    for (int32_t y = 0; y < H; y += 8) {
        int32_t ye = y + 8;
        if (ye > H) ye = H;
        int32_t t = (H > 1) ? (y * 255 / (H - 1)) : 0;
        uint8_t r = (uint8_t)(p->bg_top[0] + ((int32_t)p->bg_bot[0] - (int32_t)p->bg_top[0]) * t / 255);
        uint8_t gg = (uint8_t)(p->bg_top[1] + ((int32_t)p->bg_bot[1] - (int32_t)p->bg_top[1]) * t / 255);
        uint8_t b = (uint8_t)(p->bg_top[2] + ((int32_t)p->bg_bot[2] - (int32_t)p->bg_top[2]) * t / 255);
        gfx_draw_rectangle(g, 0, y, (uint32_t)g->width, (uint32_t)(ye - y), r, gg, b, 1);
    }
}

// 按钮：state 0=常态 1=按下 2=选中 3=关闭态（弱化）
static void flip_ui_button(Nano_GFX *g, const Flip_Ui_Palette *p,
                           int32_t x, int32_t y, int32_t w, int32_t h, int32_t r,
                           const wchar_t *label, int32_t state) {
    if (w <= 0 || h <= 0) return;
    if (state == 2) {
        flip_ui_round(g, x, y, w, h, r, p->accent, 1);
        flip_ui_text_centered(g, GFX_FONT_ALPHA_12, label, x + w / 2, y + h / 2, p->on_accent, 1);
        return;
    }
    if (state == 1) {
        flip_ui_round(g, x, y, w, h, r, p->surface_press, 1);
        flip_ui_round(g, x, y, w, h, r, p->accent, 90);
    }
    else {
        flip_ui_round(g, x, y, w, h, r, p->surface, 1);
    }
    // 1px 描边（外圈描边色 + 内圈底色，圆角一致）
    flip_ui_round(g, x, y, w, h, r, p->surface_edge, 1);
    flip_ui_round(g, x + 1, y + 1, w - 2, h - 2, r - 1,
                  (state == 1) ? p->surface_press : p->surface, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, label, x + w / 2, y + h / 2,
                          (state == 3) ? p->dim : p->text, 1);
}

// ---- 顶栏 ------------------------------------------------------------------
static void flip_ui_draw_header(Key_Event *key_event, Global_State *global_state,
                                const Flip_Ui *u, const Flip_Ui_Palette *p,
                                uint64_t elapsed_ms, int32_t running,
                                int32_t pressed) {
    Nano_GFX *g = global_state->gfx;

    // 标题
    flip_ui_text(g, GFX_FONT_ALPHA_16, L"电子沙漏", u->title_x, u->title_y, p->text, 1);

    // 状态胶囊
    int32_t chip_cy = u->chip_y + u->chip_h / 2;
    if (running) {
        flip_ui_round(g, u->chip_x, u->chip_y, u->chip_w, u->chip_h, u->chip_h / 2, p->surface, 1);
        flip_ui_round(g, u->chip_x, u->chip_y, u->chip_w, u->chip_h, u->chip_h / 2, p->surface_edge, 1);
        flip_ui_round(g, u->chip_x + 1, u->chip_y + 1, u->chip_w - 2, u->chip_h - 2,
                      u->chip_h / 2 - 1, p->surface, 1);
        flip_ui_circle(g, u->chip_x + u->chip_h / 2, chip_cy, u->chip_h / 6, p->accent, 1);
        flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"运行中",
                              u->chip_x + u->chip_w / 2 + u->chip_h / 4, chip_cy, p->dim, 1);
    }
    else {
        flip_ui_round(g, u->chip_x, u->chip_y, u->chip_w, u->chip_h, u->chip_h / 2, p->accent, 1);
        flip_ui_circle(g, u->chip_x + u->chip_h / 2, chip_cy, u->chip_h / 6, p->on_accent, 1);
        flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"已完成",
                              u->chip_x + u->chip_w / 2 + u->chip_h / 4, chip_cy, p->on_accent, 1);
    }

    // 模式切换胶囊：沙漏 ⇄ 自由流体（文字是“将要切换到的模式”）
    int32_t mode_pressed = (pressed == FLIP_UI_MODE);
    flip_ui_round(g, u->mode_x, u->mode_y, u->mode_w, u->mode_h, u->mode_r,
                  mode_pressed ? p->surface_press : p->surface, 1);
    flip_ui_round(g, u->mode_x, u->mode_y, u->mode_w, u->mode_h, u->mode_r,
                  mode_pressed ? p->accent : p->surface_edge, 1);
    flip_ui_round(g, u->mode_x + 1, u->mode_y + 1, u->mode_w - 2, u->mode_h - 2,
                  u->mode_r - 1, mode_pressed ? p->surface_press : p->surface, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, s_ui_flip_free_mode ? L"沙漏" : L"自由",
                          u->mode_x + u->mode_w / 2, u->mode_y + u->mode_h / 2, p->accent, 1);

    // 数码计时：MM:SS + 百分秒（右对齐到退出按钮左侧）
    int32_t total_sec = (int32_t)(elapsed_ms / 1000u);
    int32_t mm = total_sec / 60;
    int32_t ss = total_sec % 60;
    int32_t cs = (int32_t)(elapsed_ms % 1000u) / 10;
    wchar_t tstr[16];
    wchar_t cstr[8];
    swprintf(tstr, sizeof(tstr) / sizeof(tstr[0]), L"%02d:%02d", mm, ss);
    swprintf(cstr, sizeof(cstr) / sizeof(cstr[0]), L".%02d", cs);

    int32_t tw = 0, th = 0;
    ui_measure_7seg_string(tstr, (float)u->seg_len, (float)u->seg_thick, (float)u->seg_gap, &tw, &th);
    int32_t cw = gfx_font_measure_text(GFX_FONT_ALPHA_12, cstr);
    int32_t tx = u->timer_right - (tw + 2 * u->seg_gap + cw);
    if (tx < u->mode_x + u->mode_w + 2) tx = u->mode_x + u->mode_w + 2;
    const uint8_t *tcol = running ? p->timer : p->accent;
    ui_draw_7seg_string(key_event, global_state, tx, u->timer_y, tstr,
                        tcol[0], tcol[1], tcol[2],
                        (float)u->seg_len, (float)u->seg_thick, (float)u->seg_gap, 0,
                        &tw, &th);
    int32_t cy_top = u->timer_y + th - 13;
    flip_ui_text(g, GFX_FONT_ALPHA_12, cstr, tx + tw + 2 * u->seg_gap, cy_top, p->dim, 1);

    // 退出按钮
    int32_t exit_state = (pressed == FLIP_UI_EXIT) ? 1 : 0;
    flip_ui_round(g, u->exit_x, u->exit_y, u->exit_w, u->exit_h, u->exit_r,
                  (exit_state == 1) ? p->surface_press : p->surface, 1);
    flip_ui_round(g, u->exit_x, u->exit_y, u->exit_w, u->exit_h, u->exit_r,
                  (exit_state == 1) ? p->accent : p->surface_edge, 1);
    flip_ui_round(g, u->exit_x + 1, u->exit_y + 1, u->exit_w - 2, u->exit_h - 2,
                  u->exit_r - 1, (exit_state == 1) ? p->surface_press : p->surface, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"退出",
                          u->exit_x + u->exit_w / 2, u->exit_y + u->exit_h / 2,
                          (exit_state == 1) ? p->accent : p->text, 1);

    // 顶栏分隔线
    gfx_draw_rectangle(g, 0, u->header_h - 1, (uint32_t)u->w, 1,
                       0, 0, 0, p->separator_alpha);
}

// ---- 舞台（拟物化：金属边框 + 上下压板 + 玻璃腔体 + 跟随重力的沙池底光）-------
// 舞台在深浅两套主题下都保持深色（自发光玻璃窗）；金属/玻璃色是固定值，只有外框
// 完成态描边、瓶颈光、沙池底光跟随色彩风格。
static const uint8_t k_stage_bevel[3]    = {78, 94, 120};   // 边框倒角高光
static const uint8_t k_stage_case[3]     = {34, 43, 58};    // 壳体
static const uint8_t k_stage_plate_hi[3] = {70, 86, 112};   // 压板亮面
static const uint8_t k_stage_plate_lo[3] = {16, 21, 30};    // 压板暗面

// 平滑径向光晕（逐像素 alpha + 平方衰减）：比一圈圈同心圆更细腻，也不会有环带。
// 只在粒子绘制之前铺底（瓶颈光 / 沙池底光），面积约 1 万像素，代价可控。
static void flip_ui_radial_glow(Nano_GFX *g, int32_t cx, int32_t cy, int32_t r,
                                const uint8_t c[3], int32_t alpha_max) {
    if (r <= 1 || alpha_max <= 0) return;
    int32_t x0 = cx - r, x1 = cx + r;
    int32_t y0 = cy - r, y1 = cy + r;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 >= (int32_t)g->width) x1 = (int32_t)g->width - 1;
    if (y1 >= (int32_t)g->height) y1 = (int32_t)g->height - 1;
    int32_t r2 = r * r;
    for (int32_t y = y0; y <= y1; y++) {
        int32_t dy = y - cy;
        for (int32_t x = x0; x <= x1; x++) {
            int32_t dx = x - cx;
            int32_t d2 = dx * dx + dy * dy;
            if (d2 >= r2) continue;
            float t = 1.0f - sqrtf((float)d2) / (float)r;
            int32_t a = (int32_t)(alpha_max * t * t);
            if (a > 0) gfx_blend_pixel(g, (uint32_t)x, (uint32_t)y, c[0], c[1], c[2], (uint8_t)a);
        }
    }
}

static void flip_ui_draw_stage_frame(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                                     int32_t finished, int32_t sand_at_bottom,
                                     int32_t free_mode) {
    // 外框：1px 深边（完成态染强调色）+ 1px 倒角高光 + 壳体
    flip_ui_round(g, u->card_x - 1, u->card_y - 1, u->card_w + 2, u->card_h + 2,
                  u->card_r + 1, finished ? p->accent_dim : p->card_edge, 1);
    flip_ui_round(g, u->card_x, u->card_y, u->card_w, u->card_h, u->card_r, k_stage_bevel, 1);
    flip_ui_round(g, u->card_x + 1, u->card_y + 1, u->card_w - 2, u->card_h - 2,
                  u->card_r - 1, k_stage_case, 1);

    // 上下压板：让沙漏看起来被夹在两条金属板之间（都落在 4px 边框内，不遮流体）
    int32_t plate_x = u->card_x + u->card_r / 2 + 1;
    int32_t plate_w = u->card_w - 2 * (u->card_r / 2 + 1);
    flip_ui_round(g, plate_x, u->card_y + 1, plate_w, 2, 1, k_stage_plate_hi, 1);
    gfx_draw_rectangle(g, plate_x, u->card_y + 3, (uint32_t)plate_w, 1,
                       k_stage_plate_lo[0], k_stage_plate_lo[1], k_stage_plate_lo[2], 1);
    gfx_draw_rectangle(g, plate_x, u->card_y + u->card_h - 3, (uint32_t)plate_w, 1,
                       k_stage_plate_hi[0], k_stage_plate_hi[1], k_stage_plate_hi[2], 1);
    flip_ui_round(g, plate_x, u->card_y + u->card_h - 2, plate_w, 2, 1, k_stage_plate_lo, 1);

    // 侧边框 1px 倒角：左亮右暗（光从左上来），两端留出圆角区
    gfx_draw_rectangle(g, u->card_x + 2, u->card_y + u->card_r + 2, 1,
                       (uint32_t)(u->card_h - 2 * (u->card_r + 2)), 52, 64, 84, 1);
    gfx_draw_rectangle(g, u->card_x + u->card_w - 3, u->card_y + u->card_r + 2, 1,
                       (uint32_t)(u->card_h - 2 * (u->card_r + 2)), 12, 16, 24, 1);

    // 流体视口底色：上暗下更暗的竖向渐变（4 段）
    int32_t fh = u->fluid_h;
    for (int32_t i = 0; i < 4; i++) {
        int32_t y0 = u->fluid_y + fh * i / 4;
        int32_t y1 = u->fluid_y + fh * (i + 1) / 4;
        int32_t t = i * 85;
        uint8_t r = (uint8_t)(p->fluid_top[0] + ((int32_t)p->fluid_bot[0] - (int32_t)p->fluid_top[0]) * t / 255);
        uint8_t gg = (uint8_t)(p->fluid_top[1] + ((int32_t)p->fluid_bot[1] - (int32_t)p->fluid_top[1]) * t / 255);
        uint8_t b = (uint8_t)(p->fluid_top[2] + ((int32_t)p->fluid_bot[2] - (int32_t)p->fluid_top[2]) * t / 255);
        gfx_draw_rectangle(g, u->fluid_x, y0, (uint32_t)u->fluid_w, (uint32_t)(y1 - y0), r, gg, b, 1);
    }

    // 瓶颈暖光：只透在空腔里（沙漏实体绘制后会盖掉外围）；自由模式没有瓶颈
    int32_t gcx = u->fluid_x + u->fluid_w / 2;
    int32_t gcy = u->fluid_y + u->fluid_h / 2;
    int32_t rg = (u->fluid_w < u->fluid_h ? u->fluid_w : u->fluid_h) / 5;
    if (!free_mode) {
        flip_ui_radial_glow(g, gcx, gcy, rg, p->accent, 26);
    }

    // 沙池底光：跟随重力落在沙粒堆积的一侧（画在粒子之前 ⇒ 沙粒像浸在一团暖光里）
    int32_t pool_y = u->fluid_y + (sand_at_bottom ? (u->fluid_h * 74 / 100)
                                                   : (u->fluid_h * 26 / 100));
    int32_t pr = (u->fluid_w < u->fluid_h ? u->fluid_w : u->fluid_h) / 4;
    flip_ui_radial_glow(g, gcx, pool_y, pr, p->accent, 18);
}

// 沙漏实体：竖向 6 段环境光渐变（上亮下暗），内沿为 1.8px 玻璃辉光 + 1px 亮边（加色）。
static void flip_ui_draw_walls(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p) {
    static const uint8_t k_wall_hi[3]  = {32, 41, 56};
    static const uint8_t k_wall_lo[3]  = {19, 25, 37};
    static const uint8_t k_rim[3]      = {104, 132, 170};
    static const uint8_t k_rim_glow[3] = {34, 46, 66};
    const int32_t BANDS = 6;

    int32_t fx = u->fluid_x;
    int32_t fy = u->fluid_y;
    int32_t fw = u->fluid_w;
    int32_t fh = u->fluid_h;
    int32_t fxe = fx + fw;
    int32_t fye = fy + fh;

    // 归一化：沿用旧版沙漏边界的比例（颈口 0.469~0.556，上下半区 0.467/0.5），
    // 并把实体延伸到视口四角，保证圆角卡片内没有漏底的空隙。
    float xl = (float)fx + 0.46875f * (float)fw;
    float xr = (float)fx + 0.55625f * (float)fw;
    float yn0 = (float)fy + 0.46667f * (float)fh;
    float yn1 = (float)fy + 0.5f * (float)fh;
    int32_t ix = (int32_t)lroundf(xl);
    int32_t iy0 = (int32_t)lroundf(yn0);
    int32_t iy1 = (int32_t)lroundf(yn1);
    int32_t ix2 = (int32_t)lroundf(xr);

    // 沙漏实体的内缘 x 随 y 变化：上半段沿斜边、颈部竖直、下半段沿斜边。
    // 用 6 条水平带拼出竖向渐变（每带一个梯形 = 两个三角形），避免上下两色的接缝。
    #define FLIP_WALL_EDGE(y, is_right) \
        ((y) <= yn0 ? ((is_right) ? (float)fxe + ((float)(y) - (float)fy) / (yn0 - (float)fy) * (xr - (float)fxe) \
                                  : (float)fx  + ((float)(y) - (float)fy) / (yn0 - (float)fy) * (xl - (float)fx)) \
        : ((y) <= yn1 ? ((is_right) ? xr : xl) \
                      : ((is_right) ? xr + ((float)(y) - yn1) / ((float)fye - yn1) * ((float)fxe - xr) \
                                    : xl + ((float)(y) - yn1) / ((float)fye - yn1) * ((float)fx - xl))))
    for (int32_t b = 0; b < BANDS; b++) {
        int32_t y0 = fy + fh * b / BANDS;
        int32_t y1 = fy + fh * (b + 1) / BANDS;
        float t = ((float)b + 0.5f) / (float)BANDS;
        uint8_t cr = (uint8_t)lroundf((float)k_wall_hi[0] + ((float)k_wall_lo[0] - (float)k_wall_hi[0]) * t);
        uint8_t cg = (uint8_t)lroundf((float)k_wall_hi[1] + ((float)k_wall_lo[1] - (float)k_wall_hi[1]) * t);
        uint8_t cb = (uint8_t)lroundf((float)k_wall_hi[2] + ((float)k_wall_lo[2] - (float)k_wall_hi[2]) * t);
        int32_t lx0 = (int32_t)lroundf(FLIP_WALL_EDGE(y0, 0));
        int32_t lx1 = (int32_t)lroundf(FLIP_WALL_EDGE(y1, 0));
        int32_t rx0 = (int32_t)lroundf(FLIP_WALL_EDGE(y0, 1));
        int32_t rx1 = (int32_t)lroundf(FLIP_WALL_EDGE(y1, 1));
        gfx_draw_triangle(g, fx, y0, lx0, y0, lx1, y1, 1, cr, cg, cb, 1);
        gfx_draw_triangle(g, fx, y0, lx1, y1, fx, y1, 1, cr, cg, cb, 1);
        gfx_draw_triangle(g, fxe, y0, rx0, y0, rx1, y1, 1, cr, cg, cb, 1);
        gfx_draw_triangle(g, fxe, y0, rx1, y1, fxe, y1, 1, cr, cg, cb, 1);
    }
    #undef FLIP_WALL_EDGE

    // 玻璃内沿：1.8px 低亮度辉光 + 1px 亮边（AA 线为加色语义，天然像自发光玻璃）
    #define FLIP_RIM(x1, y1, x2, y2) do { \
        gfx_draw_line_anti_aliasing(g, (x1), (y1), (x2), (y2), 1.8f, \
            k_rim_glow[0], k_rim_glow[1], k_rim_glow[2], 1); \
        gfx_draw_line_anti_aliasing(g, (x1), (y1), (x2), (y2), 1.0f, \
            k_rim[0], k_rim[1], k_rim[2], 1); \
    } while (0)
    FLIP_RIM((float)fx, (float)fy, xl, yn0);
    FLIP_RIM(xl, yn0, xl, yn1);
    FLIP_RIM(xl, yn1, (float)fx, (float)fye);
    FLIP_RIM((float)fxe, (float)fy, xr, yn0);
    FLIP_RIM(xr, yn0, xr, yn1);
    FLIP_RIM(xr, yn1, (float)fxe, (float)fye);
    #undef FLIP_RIM

    // 颈口一线微光（提示这里是通道）
    gfx_draw_rectangle(g, ix, iy1, (uint32_t)(ix2 - ix), 1,
                       p->accent_dim[0], p->accent_dim[1], p->accent_dim[2], 110);

    // 压板内侧：顶部亮边 + 阴影（遮住粒子越界溢出的 1 像素）；底部阴影 + 暗边
    gfx_draw_rectangle(g, fx, fy, (uint32_t)fw, 1,
                       k_stage_plate_hi[0], k_stage_plate_hi[1], k_stage_plate_hi[2], 1);
    gfx_draw_rectangle(g, fx, fy + 1, (uint32_t)fw, 1, 0, 0, 0, 90);
    gfx_draw_rectangle(g, fx, fye - 2, (uint32_t)fw, 1, 0, 0, 0, 90);
    gfx_draw_rectangle(g, fx, fye - 1, (uint32_t)fw, 1,
                       k_stage_plate_lo[0], k_stage_plate_lo[1], k_stage_plate_lo[2], 1);
}

// 自由流体：矩形槽的四壁（1px 暗边 + 顶部亮边/底部阴影，遮住粒子溢出的边缘）。
static void flip_ui_draw_tank_edges(Nano_GFX *g, const Flip_Ui *u) {
    int32_t fx = u->fluid_x;
    int32_t fy = u->fluid_y;
    int32_t fw = u->fluid_w;
    int32_t fh = u->fluid_h;
    int32_t fxe = fx + fw;
    int32_t fye = fy + fh;
    // 左右/底部槽壁
    gfx_draw_rectangle(g, fx, fy + 1, 1, (uint32_t)(fh - 2),
                       k_stage_plate_lo[0], k_stage_plate_lo[1], k_stage_plate_lo[2], 1);
    gfx_draw_rectangle(g, fxe - 1, fy + 1, 1, (uint32_t)(fh - 2),
                       k_stage_plate_lo[0], k_stage_plate_lo[1], k_stage_plate_lo[2], 1);
    gfx_draw_rectangle(g, fx, fye - 1, (uint32_t)fw, 1,
                       k_stage_plate_lo[0], k_stage_plate_lo[1], k_stage_plate_lo[2], 1);
    // 顶部：亮边 + 阴影；底部：阴影 + 暗边
    gfx_draw_rectangle(g, fx, fy, (uint32_t)fw, 1,
                       k_stage_plate_hi[0], k_stage_plate_hi[1], k_stage_plate_hi[2], 1);
    gfx_draw_rectangle(g, fx, fy + 1, (uint32_t)fw, 1, 0, 0, 0, 90);
    gfx_draw_rectangle(g, fx, fye - 2, (uint32_t)fw, 1, 0, 0, 0, 90);
}

// 前玻璃：斜向反光 + 上下内阴影，把粒子与沙漏实体统一到同一块玻璃后面。
// 注：gfx_draw_triangle 的 mode>=4 在扫描线路径里按“加色”处理，故这里用低加色值
// 叠 3 层形成柔和的斜向高光；矩形则走逐像素 alpha，用于压暗的内阴影。
static void flip_ui_draw_glass(Nano_GFX *g, const Flip_Ui *u, int32_t free_mode) {
    int32_t fx = u->fluid_x;
    int32_t fy = u->fluid_y;
    int32_t fw = u->fluid_w;
    int32_t fh = u->fluid_h;
    int32_t fxe = fx + fw;
    int32_t fye = fy + fh;

    // 斜向反光：三层由宽到窄、由暗到亮（低加色，避免压过粒子）
    gfx_draw_triangle(g, fx + fw * 4 / 100, fy, fx + fw * 40 / 100, fy,
                      fx + fw * 60 / 100, fye, 1, 2, 3, 4, 3);
    gfx_draw_triangle(g, fx + fw * 4 / 100, fy, fx + fw * 60 / 100, fye,
                      fx + fw * 26 / 100, fye, 1, 2, 3, 4, 3);
    gfx_draw_triangle(g, fx + fw * 19 / 100, fy, fx + fw * 30 / 100, fy,
                      fx + fw * 51 / 100, fye, 1, 3, 3, 5, 3);
    gfx_draw_triangle(g, fx + fw * 19 / 100, fy, fx + fw * 51 / 100, fye,
                      fx + fw * 40 / 100, fye, 1, 3, 3, 5, 3);
    gfx_draw_triangle(g, fx + fw * 24 / 100, fy, fx + fw * 27 / 100, fy,
                      fx + fw * 48 / 100, fye, 1, 3, 4, 6, 3);
    gfx_draw_triangle(g, fx + fw * 24 / 100, fy, fx + fw * 48 / 100, fye,
                      fx + fw * 45 / 100, fye, 1, 3, 4, 6, 3);

    // 上下内阴影（逐像素 alpha）：玻璃凹陷的深度
    for (int32_t i = 1; i <= 4; i++) {
        uint8_t a = (uint8_t)(28 - (i - 1) * 7);
        gfx_draw_rectangle(g, fx, fy + i, (uint32_t)fw, 1, 0, 0, 0, a);
        gfx_draw_rectangle(g, fx, fye - 1 - i, (uint32_t)fw, 1, 0, 0, 0, a);
    }

    // 自由模式的矩形槽：左右内壁也补一道内阴影，强调“玻璃缸”四壁
    if (free_mode) {
        for (int32_t i = 1; i <= 3; i++) {
            uint8_t a = (uint8_t)(22 - (i - 1) * 6);
            gfx_draw_rectangle(g, fx + i, fy + 4, 1, (uint32_t)(fh - 8), 0, 0, 0, a);
            gfx_draw_rectangle(g, fxe - 1 - i, fy + 4, 1, (uint32_t)(fh - 8), 0, 0, 0, a);
        }
    }
}

// ---- 左右仪表 --------------------------------------------------------------
static void flip_ui_draw_progress_rail(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                                       float progress, int32_t remaining, int32_t grow_up) {
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"进度", u->lrail_cx, u->label_cy, p->dim, 1);

    // 竖槽
    flip_ui_round(g, u->lgauge_x, u->gauge_y, u->gauge_w, u->gauge_h, u->gauge_w / 2, p->track, 1);
    int32_t fill_h = (int32_t)lroundf(progress * (float)u->gauge_h);
    if (fill_h > 0) {
        // 起点与增长方向跟随重力：重力朝屏幕下方（沙漏正放）→ 从底部长起；
        // 重力朝屏幕上方（倒置）→ 从顶部长下来。与沙粒实际的落向一致。
        int32_t y = grow_up ? (u->gauge_y + u->gauge_h - fill_h) : u->gauge_y;
        flip_ui_round(g, u->lgauge_x, y, u->gauge_w, fill_h, u->gauge_w / 2, p->accent, 1);
    }
    // 槽内刻度（4 格，1px，很淡）
    for (int32_t i = 1; i < 4; i++) {
        int32_t y = u->gauge_y + u->gauge_h * i / 4;
        gfx_draw_rectangle(g, u->lgauge_x + 1, y, (uint32_t)(u->gauge_w - 2), 1,
                           p->bg_top[0], p->bg_top[1], p->bg_top[2], 90);
    }

    wchar_t val[8];
    swprintf(val, sizeof(val) / sizeof(val[0]), L"%d%%", (int32_t)floorf(progress * 100.0f + 0.5f));
    flip_ui_text_centered(g, GFX_FONT_ALPHA_16, val, u->lrail_cx, u->gauge_value_cy, p->text, 1);

    wchar_t rem[16];
    swprintf(rem, sizeof(rem) / sizeof(rem[0]), L"余%d", remaining);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, rem, u->lrail_cx, u->gauge_caption_cy, p->dim, 1);
}

static void flip_ui_slider_knob_y(const Flip_Ui *u, int32_t value, int32_t *out_y) {
    float t = ((float)value - 10.0f) / 90.0f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    *out_y = u->gauge_y + u->gauge_h - (int32_t)lroundf(t * (float)u->gauge_h);
}

static void flip_ui_draw_throttle_rail(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                                       int32_t value, int32_t effective, int32_t enabled,
                                       int32_t pressed) {
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"节流度", u->rrail_cx, u->label_cy, p->dim, 1);

    // 竖槽
    flip_ui_round(g, u->rgauge_x, u->gauge_y, u->gauge_w, u->gauge_h, u->gauge_w / 2, p->track, 1);

    int32_t knob_y = 0;
    flip_ui_slider_knob_y(u, value, &knob_y);
    int32_t fill_y = knob_y;
    if (fill_y < u->gauge_y) fill_y = u->gauge_y;
    int32_t fill_h = u->gauge_y + u->gauge_h - fill_y;
    if (fill_h > 0) {
        flip_ui_round(g, u->rgauge_x, fill_y, u->gauge_w, fill_h, u->gauge_w / 2,
                      enabled ? p->accent : p->accent_off, 1);
    }

    // 当前自适应节流值：槽内一道细标线（关闭时不画）
    if (enabled) {
        int32_t eff_y = 0;
        flip_ui_slider_knob_y(u, effective, &eff_y);
        gfx_draw_rectangle(g, u->rgauge_x - 2, eff_y, (uint32_t)(u->gauge_w + 4), 1,
                           p->text[0], p->text[1], p->text[2], 120);
    }

    // 推钮：横向小胶囊（比槽宽，像推子）；外圈用底色描一圈，和同色填充的轨道分开
    int32_t knob_w = u->gauge_w * 2;
    int32_t knob_h = u->knob_r * 2;
    int32_t kx = u->rrail_cx - knob_w / 2;
    int32_t ky = knob_y - knob_h / 2;
    flip_ui_round(g, kx - 1, ky - 1, knob_w + 2, knob_h + 2, knob_h / 2 + 1, p->bg_top, 1);
    flip_ui_round(g, kx, ky, knob_w, knob_h, knob_h / 2,
                  (pressed == FLIP_UI_SLIDER) ? (enabled ? p->accent : p->accent_dim)
                                              : (enabled ? p->accent : p->accent_off), 1);
    // 推钮中点一道细亮线（握持感）
    gfx_draw_rectangle(g, kx + knob_w / 2 - 3, ky + knob_h / 2, 6, 1,
                       p->on_accent[0], p->on_accent[1], p->on_accent[2],
                       enabled ? 90 : 40);

    wchar_t val[8];
    swprintf(val, sizeof(val) / sizeof(val[0]), L"%d%%", value);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_16, val, u->rrail_cx, u->gauge_value_cy,
                          enabled ? p->text : p->dim, 1);

    wchar_t cur[16];
    // 当前自适应值只写数字（不写 %）：轨道只有 44px 宽，「当前100%」会被屏幕边缘裁掉
    if (enabled) swprintf(cur, sizeof(cur) / sizeof(cur[0]), L"当前%d", effective);
    else wcsncpy(cur, L"已关闭", sizeof(cur) / sizeof(cur[0]));
    cur[sizeof(cur) / sizeof(cur[0]) - 1] = L'\0';
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, cur, u->rrail_cx, u->gauge_caption_cy,
                          enabled ? p->accent : p->dim, 1);
}

// 自由流体模式的仪表：左=粒子总数/矩形槽，右=模式/自由/无节流。
static void flip_ui_draw_free_rails(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                                    int32_t total_count) {
    // 没有竖槽/滑杆，三行文字在仪表列里垂直居中，和两侧留白平衡
    int32_t cy = u->rail_y + u->rail_h / 2;
    int32_t y_label = cy - 28;
    int32_t y_value = cy;
    int32_t y_caption = cy + 24;

    // 左：粒子数
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"粒子", u->lrail_cx, y_label, p->dim, 1);
    wchar_t cnt[16];
    swprintf(cnt, sizeof(cnt) / sizeof(cnt[0]), L"%d", total_count);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_16, cnt, u->lrail_cx, y_value, p->text, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"矩形槽", u->lrail_cx, y_caption, p->dim, 1);

    // 右：模式
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"模式", u->rrail_cx, y_label, p->dim, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_16, L"自由", u->rrail_cx, y_value, p->accent, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"无节流", u->rrail_cx, y_caption, p->dim, 1);
}

// ---- 「配色」胶囊（底部控制条；打开模态色板）--------------------------------
static void flip_ui_draw_palette_pill(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                                      int32_t pressed) {
    int32_t x = u->pal_x, y = u->btn_y, w = u->pal_w, h = u->btn_h;
    int32_t st = (pressed == FLIP_UI_PALETTE) ? 1 : 0;
    // 与普通按钮同一描边风格：外圈描边色 + 内圈底色
    flip_ui_round(g, x, y, w, h, u->btn_r, p->surface_edge, 1);
    flip_ui_round(g, x + 1, y + 1, w - 2, h - 2, u->btn_r - 1,
                  st ? p->surface_press : p->surface, 1);
    if (st) {
        flip_ui_round(g, x + 1, y + 1, w - 2, h - 2, u->btn_r - 1, p->accent, 70);
    }
    // 当前强调色圆点 + 「配色」
    int32_t group = 6 + 3 + 24;
    int32_t gx = x + (w - group) / 2;
    int32_t cy = y + h / 2;
    flip_ui_circle(g, gx + 3, cy, 3, p->accent, 1);
    flip_ui_text(g, GFX_FONT_ALPHA_12, L"配色", gx + 9, cy - 7, p->text, 1);
}

// ---- 底部控制条 ------------------------------------------------------------
static void flip_ui_draw_dock(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                              int32_t pressed, int32_t pressed_view,
                              float flow_per_sec, int32_t free_mode) {
    // 复位
    flip_ui_button(g, p, u->reset_x, u->btn_y, u->reset_w, u->btn_h, u->btn_r, L"复位",
                   (pressed == FLIP_UI_RESET) ? 1 : 0);

    // 画风：三分段控件
    int32_t gx = u->view_x;
    int32_t gy = u->btn_y;
    int32_t gw = u->view_w;
    int32_t gh = u->btn_h;
    flip_ui_round(g, gx, gy, gw, gh, u->btn_r, p->surface, 1);
    flip_ui_round(g, gx, gy, gw, gh, u->btn_r, p->surface_edge, 1);
    flip_ui_round(g, gx + 1, gy + 1, gw - 2, gh - 2, u->btn_r - 1, p->surface, 1);
    static const wchar_t *view_labels[3] = { L"粒子", L"网格", L"叠加" };
    for (int32_t i = 0; i < 3; i++) {
        int32_t cx = gx + u->view_cell_w * i;
        int32_t cw = u->view_cell_w;
        if (i == 2) cw = gx + gw - cx;
        int32_t selected = (s_ui_flip_setting_count == i);
        if (selected) {
            flip_ui_round(g, cx + 2, gy + 2, cw - 4, gh - 4, u->btn_r - 2, p->accent, 1);
        }
        else if (pressed == FLIP_UI_VIEW && pressed_view == i) {
            flip_ui_round(g, cx + 2, gy + 2, cw - 4, gh - 4, u->btn_r - 2, p->surface_press, 1);
        }
        flip_ui_text_centered(g, GFX_FONT_ALPHA_12, view_labels[i], cx + cw / 2, gy + gh / 2,
                              selected ? p->on_accent : p->dim, 1);
    }

    // 节流开关（自由流体模式下节流机制整体关闭：画成不可用指示，不响应点按）
    if (free_mode) {
        flip_ui_round(g, u->thr_x, u->btn_y, u->thr_w, u->btn_h, u->btn_r, p->surface_edge, 1);
        flip_ui_round(g, u->thr_x + 1, u->btn_y + 1, u->thr_w - 2, u->btn_h - 2,
                      u->btn_r - 1, p->surface, 1);
        flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"节流 -",
                              u->thr_x + u->thr_w / 2, u->btn_y + u->btn_h / 2, p->dim, 1);
    }
    else {
        wchar_t thr_label[16];
        if (s_ui_flip_is_throttle) wcsncpy(thr_label, L"节流 开", 16);
        else wcsncpy(thr_label, L"节流 关", 16);
        thr_label[15] = L'\0';
        int32_t thr_state;
        if (s_ui_flip_is_throttle) {
            thr_state = 2;
            if (pressed == FLIP_UI_THROTTLE) {
                // 选中态按下：用稍暗的强调色反馈
                flip_ui_round(g, u->thr_x, u->btn_y, u->thr_w, u->btn_h, u->btn_r, p->accent_dim, 1);
                flip_ui_text_centered(g, GFX_FONT_ALPHA_12, thr_label,
                                      u->thr_x + u->thr_w / 2, u->btn_y + u->btn_h / 2, p->on_accent, 1);
                thr_state = -1;
            }
        }
        else {
            thr_state = (pressed == FLIP_UI_THROTTLE) ? 1 : 3;
        }
        if (thr_state >= 0) {
            flip_ui_button(g, p, u->thr_x, u->btn_y, u->thr_w, u->btn_h, u->btn_r, thr_label, thr_state);
        }
    }

    // 配色
    flip_ui_draw_palette_pill(g, u, p, pressed);

    // 流量读数（为 0 时弱化，避免流完后仍是一个抢眼的 0/s）
    wchar_t flow_str[20];
    swprintf(flow_str, sizeof(flow_str) / sizeof(flow_str[0]), L"%.0f/s", flow_per_sec);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"流量", u->stat_cx, u->stat_y1, p->dim, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, flow_str, u->stat_cx, u->stat_y2,
                          (flow_per_sec < 0.5f) ? p->dim : p->accent, 1);
}

// ---- 配色选择弹层（模态：半透明遮罩 + 居中卡片 + 5 个大色块）-----------------
static void flip_ui_draw_picker(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                                int32_t is_light, int32_t pressed_idx) {
    // 遮罩：压暗整屏，突出模态
    gfx_draw_rectangle(g, 0, 0, (uint32_t)u->w, (uint32_t)u->h, 0, 0, 0, 110);

    // 卡片
    flip_ui_round(g, u->pick_x - 1, u->pick_y - 1, u->pick_w + 2, u->pick_h + 2,
                  u->pick_r + 1, p->card_edge, 1);
    flip_ui_round(g, u->pick_x, u->pick_y, u->pick_w, u->pick_h, u->pick_r, p->surface, 1);
    flip_ui_text(g, GFX_FONT_ALPHA_12, L"色彩风格", u->pick_title_x, u->pick_title_y, p->text, 1);

    for (int32_t i = 0; i < FLIP_COLOR_STYLES; i++) {
        int32_t sx = u->sw_x + i * (u->sw_w + u->sw_gap);
        const uint8_t *a = flip_style_accent(i, is_light);
        uint8_t on[3];
        flip_style_on_accent(a, on);
        // 选中：2px 描边环（先画大一圈的环，再画色块，留下 2px 边）
        if (i == s_ui_flip_color_style) {
            flip_ui_round(g, sx - 2, u->sw_y - 2, u->sw_w + 4, u->sw_h + 4,
                          u->sw_r + 2, p->text, 1);
        }
        flip_ui_round(g, sx, u->sw_y, u->sw_w, u->sw_h, u->sw_r, a, 1);
        if (i == pressed_idx) {
            // 按下：叠一层文字色（亮底变暗、暗底变亮）
            flip_ui_round(g, sx, u->sw_y, u->sw_w, u->sw_h, u->sw_r, on, 70);
        }
        flip_ui_text_centered(g, GFX_FONT_ALPHA_12, flip_style_names[i],
                              sx + u->sw_w / 2, u->sw_y + u->sw_h / 2, on, 1);
    }
}

// ---- 画风切换 --------------------------------------------------------------
static void flip_ui_apply_view(int32_t mode) {
    if (mode < 0) mode = 0;
    if (mode > 2) mode = 2;
    s_ui_flip_setting_count = mode;
    if (mode == FLIP_VIEW_PARTICLE) {
        s_ui_flip_show_grid = 0;
        s_ui_flip_show_particles = 1;
    }
    else if (mode == FLIP_VIEW_GRID) {
        s_ui_flip_show_grid = 1;
        s_ui_flip_show_particles = 0;
    }
    else {
        s_ui_flip_show_grid = 1;
        s_ui_flip_show_particles = 1;
    }
}

// ============================================================================
// 公开接口
// ============================================================================

void ui_app_flip_init(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    s_ui_fanqie_start_timestamp = global_state->timestamp;
    s_ui_fanqie_stop_timestamp = 0;
    s_ui_fanqie_is_running = 1;
    s_ui_fanqie_alarm_count = 0;
    s_ui_fanqie_alarm_phase = 0;
    s_ui_flip_last_upper_count = -1;
    s_ui_flip_last_upper_count_timestamp = global_state->timestamp;
    s_ui_flip_ui_pressed = FLIP_UI_NONE;
    s_ui_flip_ui_slider_active = 0;
    s_ui_flip_picker_open = 0;
    s_ui_flip_picker_pressed = FLIP_UI_NONE;
    float k = (float)(global_state->gfx->width) / (float)(global_state->gfx->height);
    // 沙漏模式：带沙漏固体边界；自由模式：只有矩形外壁
    flip_init(k, 1.0f, FLIP_RESOLUTION, global_state->timestamp,
              s_ui_flip_free_mode ? 0 : 1);
}

void ui_app_flip_render_frame(Key_Event *key_event, Global_State *global_state) {
    // 帧率节流：模拟按固定步长 dt 推进，帧率即物理流速。设备原生帧率受渲染耗时
    // 限制（ESP32/树莓派 SPI 推屏、nano_tty 终端 ANSI 渲染，均低于 30fps），表观
    // 流速正常；但高性能平台（nano_web 于 x86，主循环可达数千 FPS）若不节流，
    // 流体将以百倍于设计的速度演化（且远端只能采到极少中间步，中间过程全部丢失）。
    // 此处限制模拟/渲染最高 30fps（未到帧间隔直接返回）：原生帧率低于 30fps 的
    // 平台上本节流不触发，行为与既有版本完全一致。
    static uint64_t s_flip_last_frame_ts = 0;
    if (global_state->timestamp - s_flip_last_frame_ts < 33) {
        return;
    }
    s_flip_last_frame_ts = global_state->timestamp;

    int32_t free_mode = s_ui_flip_free_mode;

    Flip_Ui u;
    Flip_Ui_Palette pal;
    flip_ui_layout(global_state->gfx, &u);
    flip_ui_palette(global_state, &pal);

    // 获取重力方向（IMU每4帧读取一次并缓存：重力为慢变量，无需每帧I2C读取）
    static float gravity_x = 0.0f;
    static float gravity_y = -9.8f;

#ifdef IMU_ENABLED

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    static int imu_frame_div = 0;
    if ((imu_frame_div++ & 3) == 0) {
        float gx = 0.0f;
        float gy = 0.0f;
        float gz = 0.0f;
        imu_read_angle(&gx, &gy, &gz);
        gravity_x = -9.8f * gx;
        gravity_y = -9.8f * gy;
    }
#else
    // 调试打印已移除（旧版每帧 printf 会把浏览器控制台刷满，也拖慢 WASM 主循环）
    imu_read_angle(&(global_state->pitch), &(global_state->roll), &(global_state->yaw));
    gravity_x = -9.8f * sinf(global_state->roll / 180.0f * M_PI);
    gravity_y = -9.8f * cosf(global_state->roll / 180.0f * M_PI);
#endif

#endif

    float k = (float)(global_state->gfx->width) / (float)(global_state->gfx->height);

    int32_t upper_count = 0;
    int32_t lower_count = 0;

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    float dt = free_mode ? (1.6f / 60.0f)
                         : ((s_ui_flip_is_throttle) ? (0.8f / 60.0f) : (1.6f / 60.0f));
#else
    float dt = free_mode ? (1.6f / 60.0f)
                         : ((s_ui_flip_is_throttle) ? (0.6f / 60.0f) : (1.6f / 60.0f));
#endif

    // ---- 背景 + 舞台外框（用上一帧的运行态决定描边色，避免多算一次流体）----
    flip_ui_bg(global_state->gfx, &pal);
    flip_ui_draw_stage_frame(global_state->gfx, &u, &pal, !s_ui_fanqie_is_running,
                             gravity_y <= 0, free_mode);

    // ---- 流体：只画进舞台视口（视口与屏幕同宽高比 ⇒ 物理参数 k / 网格不变）----
    flip_set_color_style(s_ui_flip_color_style);   // 纯渲染配色（不影响物理）
    render_flip(
        global_state->gfx, u.fluid_x, u.fluid_y, u.fluid_w, u.fluid_h,
        k, 1.0f, FLIP_RESOLUTION,     /* pool_width, pool_height, resolution */
        gravity_x, gravity_y,    /* gravity_x, gravity_y */
        dt,    /* dt */
        0.8f,            /* flip_ratio */
        20, 2,           /* num_pressure_iters, num_particle_iters */
        1.0f,            /* over_relaxation */
        1, 1,            /* compensate_drift, separate_particles */
        s_ui_flip_show_particles, s_ui_flip_show_grid,
        (free_mode || !s_ui_flip_is_throttle) ? 0 : s_ui_flip_throttle,
        &upper_count, &lower_count
    );

    // 沙漏实体 / 矩形槽四壁盖在粒子之上（自由模式没有沙漏壁与节流）
    if (free_mode) {
        flip_ui_draw_tank_edges(global_state->gfx, &u);
    }
    else {
        flip_ui_draw_walls(global_state->gfx, &u, &pal);
    }
    // 前玻璃：斜向反光 + 内阴影，让粒子与实体看起来在同一块玻璃后面
    flip_ui_draw_glass(global_state->gfx, &u, free_mode);

    // ---- 沙漏重置判定（逻辑与旧版一致）；自由流体模式没有“流完”，计时器只做秒表 ----
    if (free_mode) {
        s_ui_fanqie_stop_timestamp = 0;
        s_ui_fanqie_is_running = 1;
    }
    else if ((lower_count < 1 && gravity_y < 0) || (upper_count < 1 && gravity_y > 0)) {
        s_ui_fanqie_start_timestamp = global_state->timestamp;
        s_ui_fanqie_stop_timestamp = 0;
        s_ui_fanqie_is_running = 1;
        s_ui_fanqie_alarm_count = 0;
    }
    else if ((upper_count < 1 && gravity_y < 0) || (lower_count < 1 && gravity_y > 0)) {
        if (!s_ui_fanqie_stop_timestamp) {
            s_ui_fanqie_stop_timestamp = global_state->timestamp;
        }
        s_ui_fanqie_is_running = 0;
    }

    // ---- 计时 ----
    uint64_t current_timestamp = global_state->timestamp;
    if (!s_ui_fanqie_is_running) {
        current_timestamp = s_ui_fanqie_stop_timestamp;
    }
    if (current_timestamp < s_ui_fanqie_start_timestamp) {
        current_timestamp = s_ui_fanqie_start_timestamp;
    }
    uint64_t elapsed_ms = current_timestamp - s_ui_fanqie_start_timestamp;

    // ---- 每1000ms统计一次粒子流量 ----
    // 首拍只建立基线（last=-1 哨兵）：否则「进入/复位后的第一帧」会把满池粒子数除以
    // 几毫秒的 dt，显示出一个几十万/秒的假流量尖峰。
    static float particle_flow_per_sec = 0.0f;
    if (s_ui_flip_last_upper_count < 0) {
        particle_flow_per_sec = 0.0f;
        s_ui_flip_last_upper_count = upper_count;
        s_ui_flip_last_upper_count_timestamp = global_state->timestamp;
    }
    else if (global_state->timestamp - s_ui_flip_last_upper_count_timestamp >= 1000) {
        uint64_t dt_ms = global_state->timestamp - s_ui_flip_last_upper_count_timestamp;
        if (dt_ms > 0) {
            particle_flow_per_sec = (float)(upper_count - s_ui_flip_last_upper_count) / (float)dt_ms * 1000;
        }
        s_ui_flip_last_upper_count = upper_count;
        s_ui_flip_last_upper_count_timestamp = global_state->timestamp;
    }

    // ---- 沙漏进度：相对重力方向，已落下的沙量占比 ----
    int32_t total_count = upper_count + lower_count;
    float hourglass_progress = 0.0f;
    if (total_count > 0) {
        hourglass_progress = (float)((gravity_y <= 0) ? lower_count : upper_count) / (float)total_count;
    }
    int32_t remaining_count = (gravity_y <= 0) ? upper_count : lower_count;

    // 根据沙漏进度调整节流度，避免来自上方的压力过小时，出现几乎不往下流的问题
    // （自由模式关闭全部节流机制）
    if (free_mode) {
        s_ui_flip_throttle = 0;
    }
    else {
        s_ui_flip_throttle = (int32_t)roundf((1.0f - (float)s_ui_flip_init_throttle) * hourglass_progress * hourglass_progress
                                             + (float)s_ui_flip_init_throttle);
    }

    // ---- 到时提醒（非阻塞状态机：跨帧推进震动节奏，不阻塞渲染任务）----
    if (!free_mode && !s_ui_fanqie_is_running && s_ui_flip_is_throttle && s_ui_fanqie_alarm_count < 6) {
        // 门槛：仅真实计时（运行时长≥3s）结束后才报警，杜绝“进入即误判流完”的伪报警
        if (s_ui_fanqie_stop_timestamp >= s_ui_fanqie_start_timestamp &&
            s_ui_fanqie_stop_timestamp - s_ui_fanqie_start_timestamp >= 3000) {
            uint64_t now = global_state->timestamp;
            if (s_ui_fanqie_alarm_phase == 0) {           // 开始一段鸣震
                set_vibration(222);
                s_ui_fanqie_alarm_phase_ts = now;
                s_ui_fanqie_alarm_phase = 1;
            }
            else if (s_ui_fanqie_alarm_phase == 1 && now - s_ui_fanqie_alarm_phase_ts >= 600) {
                set_vibration(0);                          // 鸣震 600ms 后进入静默段
                s_ui_fanqie_alarm_phase_ts = now;
                s_ui_fanqie_alarm_phase = 2;
            }
            else if (s_ui_fanqie_alarm_phase == 2 && now - s_ui_fanqie_alarm_phase_ts >= 600) {
                s_ui_fanqie_alarm_count++;                 // 静默 600ms 后完成一个周期
                s_ui_fanqie_alarm_phase = 0;
            }
        }
    }
    else if (s_ui_fanqie_alarm_phase != 0) {
        // 报警被中断（重新计时/关闭节流/报警完成）：确保马达关闭、状态机复位
        set_vibration(0);
        s_ui_fanqie_alarm_phase = 0;
    }

    // ---- chrome：顶栏 / 左右仪表 / 底部控制条 ----
    flip_ui_draw_header(key_event, global_state, &u, &pal, elapsed_ms,
                        s_ui_fanqie_is_running, s_ui_flip_ui_pressed);
    if (free_mode) {
        flip_ui_draw_free_rails(global_state->gfx, &u, &pal, total_count);
    }
    else {
        // 进度槽的起点/方向跟随重力（gravity_y < 0 = 重力朝屏幕下方 = 正放）
        flip_ui_draw_progress_rail(global_state->gfx, &u, &pal, hourglass_progress, remaining_count,
                                   gravity_y <= 0);
        flip_ui_draw_throttle_rail(global_state->gfx, &u, &pal, s_ui_flip_init_throttle,
                                   s_ui_flip_throttle, s_ui_flip_is_throttle, s_ui_flip_ui_pressed);
    }
    flip_ui_draw_dock(global_state->gfx, &u, &pal, s_ui_flip_ui_pressed,
                      s_ui_flip_ui_pressed_view, fabsf(particle_flow_per_sec), free_mode);

    // 配色选择弹层（模态）：画在全部 chrome 之上
    if (s_ui_flip_picker_open) {
        flip_ui_draw_picker(global_state->gfx, &u, &pal,
                            global_state->ui_color_style == UI_COLOR_LIGHT,
                            s_ui_flip_picker_pressed);
    }

    gfx_refresh(global_state->gfx);
}

// ---- 触屏命中判定（画风分段控件返回整组，另用 view_index 区分格子）---------
static int32_t flip_ui_hit(const Flip_Ui *u, int32_t x, int32_t y) {
    if (x >= u->exit_x && x < u->exit_x + u->exit_w &&
        y >= u->exit_y && y < u->exit_y + u->exit_h) {
        return FLIP_UI_EXIT;
    }
    // 顶栏模式切换胶囊（沙漏 ⇄ 自由流体）
    if (x >= u->mode_x && x < u->mode_x + u->mode_w &&
        y >= u->mode_y && y < u->mode_y + u->mode_h) {
        return FLIP_UI_MODE;
    }
    if (y >= u->btn_y && y < u->btn_y + u->btn_h) {
        if (x >= u->reset_x && x < u->reset_x + u->reset_w) return FLIP_UI_RESET;
        if (x >= u->view_x && x < u->view_x + u->view_w) return FLIP_UI_VIEW;
        // 自由流体模式关闭节流机制：节流开关不响应
        if (!s_ui_flip_free_mode && x >= u->thr_x && x < u->thr_x + u->thr_w) return FLIP_UI_THROTTLE;
        if (x >= u->pal_x && x < u->pal_x + u->pal_w) return FLIP_UI_PALETTE;
    }
    // 右仪表：轨道上下各放宽一个推钮半径，便于点按/拖动；标签与读数区不响应
    // （自由模式右栏是模式信息，没有节流滑杆）
    if (!s_ui_flip_free_mode &&
        x >= u->rrail_x && x < u->rrail_x + u->rrail_w &&
        y >= u->gauge_y - u->knob_r && y <= u->gauge_y + u->gauge_h + u->knob_r) {
        return FLIP_UI_SLIDER;
    }
    return FLIP_UI_NONE;
}

static int32_t flip_ui_view_index(const Flip_Ui *u, int32_t x) {
    if (x < u->view_x) return 0;
    int32_t rel = x - u->view_x;
    if (rel >= u->view_w) return 2;
    int32_t idx = rel / (u->view_cell_w > 0 ? u->view_cell_w : 1);
    if (idx < 0) idx = 0;
    if (idx > 2) idx = 2;
    return idx;
}

// 配色色块命中（返回 0..4，未命中 -1）；与弹层绘制共用同一份几何
static int32_t flip_ui_swatch_hit(const Flip_Ui *u, int32_t x, int32_t y) {
    if (y < u->sw_y || y >= u->sw_y + u->sw_h) {
        return FLIP_UI_NONE;
    }
    for (int32_t i = 0; i < FLIP_COLOR_STYLES; i++) {
        int32_t sx = u->sw_x + i * (u->sw_w + u->sw_gap);
        if (x >= sx && x < sx + u->sw_w) return i;
    }
    return FLIP_UI_NONE;
}

static int32_t flip_ui_in_rect(int32_t x, int32_t y, int32_t w, int32_t h,
                               int32_t px, int32_t py) {
    return (px >= x && px < x + w && py >= y && py < y + h);
}

static void flip_ui_slider_set(const Flip_Ui *u, int32_t y) {
    float t = (float)(u->gauge_y + u->gauge_h - y) / (float)(u->gauge_h > 0 ? u->gauge_h : 1);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    int32_t v = 10 + (int32_t)lroundf(t * 90.0f);
    if (v < 10) v = 10;
    if (v > 100) v = 100;
    s_ui_flip_init_throttle = v;
}

void ui_app_flip_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 渲染帧里可能已经切走（退出按钮），此时不再处理输入
    if (global_state->STATE != STATE_FLIP) {
        return;
    }

    Flip_Ui u;
    flip_ui_layout(global_state->gfx, &u);

    // ---- 配色选择弹层打开时：模态，先消费触屏/ESC，普通控件一律不响应 ----
    if (s_ui_flip_picker_open) {
        if (key_event->touch_edge & TOUCH_EDGE_DOWN) {
            s_ui_flip_picker_pressed = flip_ui_swatch_hit(&u, key_event->touch_down_x, key_event->touch_down_y);
        }
        if (key_event->touch_edge & TOUCH_EDGE_UP) {
            int32_t idx = s_ui_flip_picker_pressed;
            int32_t rel = flip_ui_swatch_hit(&u, key_event->touch_x, key_event->touch_y);
            s_ui_flip_picker_pressed = FLIP_UI_NONE;
            if (idx >= 0 && idx == rel) {
                s_ui_flip_color_style = idx;     // 选中即生效并关闭
                s_ui_flip_picker_open = 0;
            }
            else if (!flip_ui_in_rect(u.pick_x, u.pick_y, u.pick_w, u.pick_h,
                                      key_event->touch_x, key_event->touch_y)) {
                s_ui_flip_picker_open = 0;       // 点卡片外：关闭，不改设置
            }
        }
        if (key_event->is_soft_key == 0 &&
            (key_event->key_edge == -1 || key_event->key_edge == -2) &&
            key_event->key_code == NANO_KEY_esc) {
            s_ui_flip_picker_open = 0;           // ESC 先关弹层，不退出应用
        }
        return;
    }

    // ---- 触屏：按下点锁存控件，松手沿结算（同 ui_water.c / ui_musicbox.c 范式）----
    if (key_event->touch_edge & TOUCH_EDGE_DOWN) {
        int32_t hit = flip_ui_hit(&u, key_event->touch_down_x, key_event->touch_down_y);
        s_ui_flip_ui_pressed = hit;
        s_ui_flip_ui_slider_active = 0;
        if (hit == FLIP_UI_VIEW) {
            s_ui_flip_ui_pressed_view = flip_ui_view_index(&u, key_event->touch_down_x);
        }
        else if (hit == FLIP_UI_SLIDER) {
            s_ui_flip_ui_slider_active = 1;
            flip_ui_slider_set(&u, key_event->touch_down_y);
        }
    }
    if (s_ui_flip_ui_slider_active && key_event->is_touching) {
        flip_ui_slider_set(&u, key_event->touch_y);
    }
    if (key_event->touch_edge & TOUCH_EDGE_UP) {
        int32_t hit = s_ui_flip_ui_pressed;
        if (hit == FLIP_UI_SLIDER) {
            s_ui_flip_ui_slider_active = 0;
        }
        else if (hit != FLIP_UI_NONE) {
            int32_t rel_hit = flip_ui_hit(&u, key_event->touch_x, key_event->touch_y);
            int32_t same = (rel_hit == hit);
            if (hit == FLIP_UI_VIEW) {
                same = same && (flip_ui_view_index(&u, key_event->touch_x) == s_ui_flip_ui_pressed_view);
            }
            if (same) {
                if (hit == FLIP_UI_EXIT) {
                    s_ui_flip_ui_pressed = FLIP_UI_NONE;
                    global_state->STATE = STATE_MAIN_MENU;
                    return;
                }
                else if (hit == FLIP_UI_RESET) {
                    ui_app_flip_init(key_event, global_state);
                }
                else if (hit == FLIP_UI_VIEW) {
                    flip_ui_apply_view(s_ui_flip_ui_pressed_view);
                }
                else if (hit == FLIP_UI_THROTTLE) {
                    s_ui_flip_is_throttle = s_ui_flip_is_throttle ? 0 : 1;
                }
                else if (hit == FLIP_UI_PALETTE) {
                    s_ui_flip_picker_open = 1;
                    s_ui_flip_picker_pressed = FLIP_UI_NONE;
                }
                else if (hit == FLIP_UI_MODE) {
                    s_ui_flip_free_mode = s_ui_flip_free_mode ? 0 : 1;
                    ui_app_flip_init(key_event, global_state);   // 固体边界变了，重初始化流体
                }
            }
        }
        s_ui_flip_ui_pressed = FLIP_UI_NONE;
        s_ui_flip_ui_slider_active = 0;
    }

    // ---- 实体键盘兜底（触屏派生的软按键一律不响应：is_soft_key == 0）----
    if (key_event->is_soft_key != 0) {
        return;
    }
    int32_t edge = (key_event->key_edge == -1 || key_event->key_edge == -2);
    if (edge && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_MAIN_MENU;
    }
    else if (edge && key_event->key_code == NANO_KEY_left) {
        // 与旧版一致：循环切换画风（粒子 → 网格 → 叠加）
        flip_ui_apply_view((s_ui_flip_setting_count + 1) % 3);
    }
    else if (edge && key_event->key_code == NANO_KEY_right) {
        flip_ui_apply_view((s_ui_flip_setting_count + 2) % 3);
    }
    else if (edge && key_event->key_code == NANO_KEY_1) {
        if (!s_ui_flip_free_mode) s_ui_flip_is_throttle = s_ui_flip_is_throttle ? 0 : 1;
    }
    else if (edge && key_event->key_code == NANO_KEY_2) {
        s_ui_flip_free_mode = s_ui_flip_free_mode ? 0 : 1;
        ui_app_flip_init(key_event, global_state);
    }
    else if (edge && key_event->key_code == NANO_KEY_ctrl) {
        s_ui_flip_init_throttle += 10;
        if (s_ui_flip_init_throttle > 100) {
            s_ui_flip_init_throttle = 10;
        }
    }
    else if (edge && key_event->key_code == NANO_KEY_enter) {
        ui_app_flip_init(key_event, global_state);
    }
}


// ===============================================================================
// 玲珑天象仪
// ===============================================================================

static uint64_t linglong_first_call_timestamp = 0;
static uint32_t linglong_last_day = 0;
static uint32_t linglong_sunrise_time[2] = {0, 0}; // hour, minute
static uint32_t linglong_sunset_time[2] = {0, 0}; // hour, minute

static int32_t linglong_timemachine_running_state = 2; // 0-停止；1-时光机运行；2-实时
static int32_t linglong_timemachine_speed = 0; // 时光机速度，正数为未来，负数为过去，单位秒
static uint64_t linglong_timemachine_start_timestamp = 0;
static int32_t linglong_paused_resume_state = 2; // 暂停前的运行状态（1-时光机；2-实时），供“运行/暂停”按钮恢复

static int32_t linglong_state = 0; // 玲珑仪UI状态

#define LL_STATE_SKY (0)
#define LL_STATE_SETTING (1)
#define LL_STATE_SETTING_CALLBACK (2)


void ui_app_linglong_init(Key_Event *key_event, Global_State *global_state) {
    linglong_state = LL_STATE_SKY;
    linglong_paused_resume_state = 2;
    ui_linglong_ctrl_init();
    s_linglong_touch_owned = 1; // 主天空界面：触屏归显式控件，抑制宫格软键
}

// 大气散射档位循环：屏蔽「简化模型(1)」「一次散射(2)」——两者分别被二次散射(3)与
// 海洋天空(5)完全覆盖（观感更差且没有独占能力），保留渲染实现但不再出现在选项里。
// 顺序：关(0) → 二次散射(3) → 体积云(4) → 海洋天空(5) → 关。
// 当前值若落在被屏蔽档位（手改配置/旧存档），下一次点击落到二次散射。
static int32_t ui_app_linglong_next_sky_model(int32_t cur) {
    static const int32_t kOrder[] = { 0, 3, 4, 5 };
    const int32_t n = (int32_t)(sizeof(kOrder) / sizeof(kOrder[0]));
    for (int32_t i = 0; i < n; i++) {
        if (kOrder[i] == cur) return kOrder[(i + 1) % n];
    }
    return 3;
}

// 云参数现值名称与控制（设置界面第 1 行前三格；原主控制面板左列）
static const wchar_t *ui_app_linglong_cloud_layer_name(Linglong_Config *cfg) {
    int32_t n = 0;
    const Linglong_Cloud_Option *opt = linglong_cloud_layer_options(&n);
    if (n <= 0) return L"关";
    for (int32_t i = 0; i < n; i++) {
        if (opt[i].value == cfg->cloud_layer_mask) return opt[i].name;
    }
    return opt[0].name;
}

static const wchar_t *ui_app_linglong_cloud_precision_name(Linglong_Config *cfg) {
    int32_t n = 0;
    const Linglong_Cloud_Option *opt = linglong_cloud_precision_options(&n);
    int32_t idx = cfg->cloud_precision_level;
    if (idx < 0 || idx >= n) idx = 0;
    return (n > 0) ? opt[idx].name : L"中";
}

static void ui_app_linglong_cloud_next_coverage(Global_State *global_state) {
    int32_t n = ui_cloud_coverage_level_num();
    if (n < 1) n = 1;
    global_state->linglong_cfg->cloud_coverage_level =
        (global_state->linglong_cfg->cloud_coverage_level + 1) % n;
}

static void ui_app_linglong_cloud_next_layer(Global_State *global_state) {
    int32_t n = 0;
    const Linglong_Cloud_Option *opt = linglong_cloud_layer_options(&n);
    int32_t idx = 0;
    for (int32_t i = 0; i < n; i++) {
        if (opt[i].value == global_state->linglong_cfg->cloud_layer_mask) { idx = i; break; }
    }
    idx = (n > 0) ? (idx + 1) % n : 0;
    if (n > 0) global_state->linglong_cfg->cloud_layer_mask = opt[idx].value;
}

static void ui_app_linglong_cloud_next_precision(Global_State *global_state) {
    int32_t n = 0;
    (void)linglong_cloud_precision_options(&n);
    if (n < 1) n = 1;
    global_state->linglong_cfg->cloud_precision_level =
        (global_state->linglong_cfg->cloud_precision_level + 1) % n;
}

// 设置界面剩余功能格映射：返回现有处理链使用的键码（调用前会置 Ctrl 激活）。
// 第 1 行前三格（云量/云类/云精度）由调用方直接循环配置，不合成按键。
static int32_t ui_app_linglong_setting_cell_key(int32_t cell) {
    switch (cell) {
        case 3:  return NANO_KEY_4;    // 黄道
        case 4:  return NANO_KEY_ctrl; // 返回
        case 7:  return NANO_KEY_5;    // 天体名称
        case 8:  return NANO_KEY_9;    // 平滑滤波
        case 9:  return NANO_KEY_0;    // 跟踪太阳
        default: return NANO_KEY_IDLE;
    }
}

// 填设置界面 5x4 单元格：标签 / 现值 / 现值颜色；返回按钮特殊样式，其余空位保留备用。
static void ui_app_linglong_setting_fill_cells(Global_State *global_state, UiLinglongSettingCell *cells) {
    Linglong_Config *cfg = global_state->linglong_cfg;

    for (int32_t i = 0; i < UI_LL_SETTING_COUNT; i++) {
        cells[i].label = NULL;
        cells[i].value = NULL;
        cells[i].value_r = 224;
        cells[i].value_g = 224;
        cells[i].value_b = 224;
        cells[i].is_special = 0;
    }

    // 第 1 行：云量、云类、云精度、黄道、返回
    // 云参数只在 sky_model==4（体积云）时影响画面；其它档位现值画灰但可预调
    // （与旧控制面板云参数按钮“不吃参数时画暗”的语义一致）。
    const int32_t cloud_active = (cfg->sky_model == 4);
    const uint8_t cloud_r = cloud_active ? 0 : 168;
    const uint8_t cloud_g = cloud_active ? 255 : 176;
    const uint8_t cloud_b = cloud_active ? 255 : 190;

    cells[0].label = L"云量";
    cells[0].value = ui_cloud_coverage_level_name(cfg->cloud_coverage_level);
    cells[0].value_r = cloud_r; cells[0].value_g = cloud_g; cells[0].value_b = cloud_b;

    cells[1].label = L"云类";
    cells[1].value = ui_app_linglong_cloud_layer_name(cfg);
    cells[1].value_r = cloud_r; cells[1].value_g = cloud_g; cells[1].value_b = cloud_b;

    cells[2].label = L"云精度";
    cells[2].value = ui_app_linglong_cloud_precision_name(cfg);
    cells[2].value_r = cloud_r; cells[2].value_g = cloud_g; cells[2].value_b = cloud_b;

    cells[3].label = L"黄道";
    if (cfg->enable_ecliptic_circle == 0) {
        cells[3].value = L"关";
        cells[3].value_r = 222; cells[3].value_g = 0; cells[3].value_b = 0;
    }
    else {
        cells[3].value = L"开";
        cells[3].value_r = 0; cells[3].value_g = 255; cells[3].value_b = 0;
    }

    cells[4].label = L"返回";
    cells[4].is_special = 1;

    // 第 2 行：前两格空出（原大气散射/地景已移到控制面板），后三格保留
    cells[7].label = L"天体名称";
    switch (cfg->enable_star_name) {
        case 0:
            cells[7].value = L"关";
            cells[7].value_r = 222; cells[7].value_g = 0; cells[7].value_b = 0;
            break;
        case 1:
            cells[7].value = L"仅恒星";
            cells[7].value_r = 0; cells[7].value_g = 255; cells[7].value_b = 255;
            break;
        case 2:
            cells[7].value = L"仅行星";
            cells[7].value_r = 0; cells[7].value_g = 255; cells[7].value_b = 255;
            break;
        case 3:
            cells[7].value = L"全部显示";
            cells[7].value_r = 0; cells[7].value_g = 255; cells[7].value_b = 255;
            break;
        default: break;
    }

    cells[8].label = L"平滑滤波";
    if (cfg->enable_opt_bilinear == 0) {
        cells[8].value = L"关";
        cells[8].value_r = 222; cells[8].value_g = 0; cells[8].value_b = 0;
    }
    else {
        cells[8].value = L"开";
        cells[8].value_r = 0; cells[8].value_g = 255; cells[8].value_b = 0;
    }

    cells[9].label = L"跟踪太阳";
    if (cfg->enable_tracking_sun == 0) {
        cells[9].value = L"关";
        cells[9].value_r = 222; cells[9].value_g = 0; cells[9].value_b = 0;
    }
    else {
        cells[9].value = L"开";
        cells[9].value_r = 0; cells[9].value_g = 255; cells[9].value_b = 0;
    }

    // 第 3 行第 2~4 格：海洋参数（观察高度 / 涌浪波高 / 涌浪波长）。
    // 观察高度对 sky_model==4（体积云相机海拔）/5 或 landscape_index==3 都有效；
    // 涌浪波高/波长只对海洋有效。不影响的档位现值画灰但可预调。
    {
        const int32_t height_active = (cfg->sky_model == 4 || cfg->sky_model == 5 ||
                                       cfg->landscape_index == 3);
        const int32_t ocean_active  = (cfg->sky_model == 5 || cfg->landscape_index == 3);
        int32_t n = 0;
        const Linglong_Ocean_Option *opt;
        struct { int32_t cell; int32_t level; const Linglong_Ocean_Option *(*get)(int32_t *); } m[3] = {
            { 11, cfg->ocean_height_level,    linglong_ocean_height_options },
            { 12, cfg->ocean_swell_hs_level,  linglong_ocean_swell_hs_options },
            { 13, cfg->ocean_swell_len_level, linglong_ocean_swell_len_options },
        };
        static const wchar_t *labels[3] = { L"观察高度", L"涌浪波高", L"涌浪波长" };
        static wchar_t s_alt_text[24];
        for (int32_t i = 0; i < 3; i++) {
            opt = m[i].get(&n);
            if (opt == NULL || n <= 0) continue;
            int32_t lv = m[i].level;
            if (lv < 0 || lv >= n) lv = 0;
            cells[m[i].cell].label = labels[i];
            if (m[i].cell == 11) {
                // 观察高度显示真实海拔（漫游时是连续量），档位只作为快捷跳转
                const double alt = (double)cfg->obs_y;
                if (alt < 1000.0) swprintf(s_alt_text, 24, L"%.0fm", alt);
                else swprintf(s_alt_text, 24, L"%.1fkm", alt / 1000.0);
                cells[m[i].cell].value = s_alt_text;
            }
            else {
                cells[m[i].cell].value = opt[lv].name;
            }
            if ((m[i].cell == 11) ? height_active : ocean_active) {
                cells[m[i].cell].value_r = 0; cells[m[i].cell].value_g = 200; cells[m[i].cell].value_b = 255;
            }
            else {
                cells[m[i].cell].value_r = 168; cells[m[i].cell].value_g = 176; cells[m[i].cell].value_b = 190;
            }
        }
    }

    // 第 3 行首格：岛屿地形开关（landscape_index==3 时生效）
    {
        static wchar_t s_terr_prog[24];
        const int32_t terr_active = (cfg->landscape_index == 3);
        const uint8_t tr = terr_active ? 0 : 168;
        const uint8_t tg = terr_active ? 200 : 176;
        const uint8_t tb = terr_active ? 255 : 190;

        cells[10].label = L"岛屿地形";
        cells[10].value = (cfg->terrain_enable) ? L"开" : L"关";
        if (cfg->terrain_enable) { cells[10].value_r = 0; cells[10].value_g = 255; cells[10].value_b = 0; }
        else { cells[10].value_r = 222; cells[10].value_g = 0; cells[10].value_b = 0; }
        if (!terr_active) { cells[10].value_r = tr; cells[10].value_g = tg; cells[10].value_b = tb; }

        cells[14].label = L"地形精度";
        {
            static const wchar_t *names[3] = { L"省电", L"标准", L"精细" };
            int32_t lv = cfg->terrain_quality;
            if (lv < 0 || lv > 2) lv = 1;
            cells[14].value = names[lv];
        }
        cells[14].value_r = tr; cells[14].value_g = tg; cells[14].value_b = tb;

        cells[15].label = L"换一座岛";
        {
            // 生成中显示进度，就绪后显示当前种子的低 4 位（便于复现同一组岛）
            int32_t prog = linglong_ocean_terrain_progress();
            if (prog < 100) {
                swprintf(s_terr_prog, 24, L"%d%%", prog);
            }
            else {
                swprintf(s_terr_prog, 24, L"#%04u", (unsigned)(cfg->terrain_seed % 10000u));
            }
            cells[15].value = s_terr_prog;
        }
        cells[15].value_r = tr; cells[15].value_g = tg; cells[15].value_b = tb;

        cells[16].label = L"地形网格";
        cells[16].value = (cfg->terrain_res >= 1024) ? L"1024" : L"512";
        cells[16].value_r = tr; cells[16].value_g = tg; cells[16].value_b = tb;

        cells[17].label = L"漫游速度";
        {
            static const wchar_t *spd[3] = { L"10m/s", L"100m/s", L"1km/s" };
            int32_t lv = cfg->roam_speed_level;
            if (lv < 0 || lv > 2) lv = 1;
            cells[17].value = spd[lv];
        }
        cells[17].value_r = 0; cells[17].value_g = 200; cells[17].value_b = 255;

        cells[18].label = L"观察位置";
        {
            static wchar_t s_pos[32];
            swprintf(s_pos, 32, L"%+.1f/%+.1fkm", (double)cfg->obs_x / 1000.0, (double)cfg->obs_z / 1000.0);
            cells[18].value = s_pos;
        }
        cells[18].value_r = 200; cells[18].value_g = 220; cells[18].value_b = 240;

        cells[19].label = L"漫游";
        cells[19].value = (cfg->roam_enable) ? L"开" : L"关";
        if (cfg->roam_enable) { cells[19].value_r = 0; cells[19].value_g = 255; cells[19].value_b = 0; }
        else { cells[19].value_r = 222; cells[19].value_g = 0; cells[19].value_b = 0; }
    }
}

void ui_app_linglong_setting_draw(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    // 不清屏：设置界面直接叠加在每帧新绘制的天象仪主界面上（与控制面板同一做法），
    // 5x4 按钮使用与控制按钮一致的圆角半透明样式。
    UiLinglongSettingCell cells[UI_LL_SETTING_COUNT];
    ui_app_linglong_setting_fill_cells(global_state, cells);
    ui_linglong_ctrl_draw_settings(global_state->llgfx, cells);
}




// ===============================================================================
// 漫游（观察者水平/垂直移动）
// ===============================================================================
// 方向来源有两路，最终合成一个位掩码后按 dt 积分（与帧率无关）：
//   · 触屏移动 pad（控制面板底部，长按连续移动）——每帧直接问当前按住的方向；
//   · 物理键/键盘（上/下/左/右、W/A/S/D、Q/E）——按键事件是边沿触发（松手沿 -1、
//     长按重复 -2、按住 0），这里把“最近一次事件”当作按住状态，180 ms 无事件即释放，
//     因而对任意重复频率都稳定，不会因长按重复而超速。
static int32_t  s_roam_key_mask = 0;
static uint64_t s_roam_key_ts[6] = { 0, 0, 0, 0, 0, 0 };
static uint64_t s_roam_last_ms = 0;

static void ui_app_linglong_roam_key_hold(int32_t bit) {
    s_roam_key_mask |= bit;
    int32_t idx = 0;
    while ((1 << idx) != bit && idx < 6) idx++;
    if (idx < 6) s_roam_key_ts[idx] = get_timestamp_in_ms();
}

static int32_t ui_app_linglong_roam_key_mask(void) {
    const uint64_t now = get_timestamp_in_ms();
    int32_t mask = 0;
    for (int32_t i = 0; i < 6; i++) {
        if (s_roam_key_mask & (1 << i)) {
            if (now - s_roam_key_ts[i] > 180) s_roam_key_mask &= ~(1 << i);
            else mask |= (1 << i);
        }
    }
    return mask;
}

// 每帧推进观察者位置（在 render_sky 之前调用）
static void ui_app_linglong_roam_step(Key_Event *key_event, Global_State *global_state) {
    Linglong_Config *c = global_state->linglong_cfg;
    const uint64_t now = get_timestamp_in_ms();
    double dt = (s_roam_last_ms == 0) ? 0.0 : (double)(now - s_roam_last_ms) / 1000.0;
    s_roam_last_ms = now;
    if (dt < 0.0) dt = 0.0;
    if (dt > 0.1) dt = 0.1;

    // 触屏 pad（只有漫游开启且面板可见才有值）
    int32_t mask = ui_linglong_ctrl_roam_held(
        key_event->is_touching, key_event->touch_x, key_event->touch_y);
    // 物理键（同样只在漫游开启时生效；否则方向键仍是时光机快退/快进）
    if (c->roam_enable) mask |= ui_app_linglong_roam_key_mask();
    if (mask == 0 || dt <= 0.0) return;

    static const float S_SPEED[3] = { 10.0f, 100.0f, 1000.0f };
    int32_t lv = c->roam_speed_level;
    if (lv < 0 || lv > 2) lv = 1;
    const double speed = (double)S_SPEED[lv];

    // 前进方向 = 视线方向的水平投影（罗盘方位：北 = +z，东 = +x）
    const double azi = (double)c->view_azi * M_PI / 180.0;
    const double fx = sin(azi), fz = cos(azi);
    const double rx = cos(azi), rz = -sin(azi);
    double mx = 0.0, mz = 0.0;
    if (mask & UI_LL_ROAM_FWD)   { mx += fx; mz += fz; }
    if (mask & UI_LL_ROAM_BACK)  { mx -= fx; mz -= fz; }
    if (mask & UI_LL_ROAM_RIGHT) { mx += rx; mz += rz; }
    if (mask & UI_LL_ROAM_LEFT)  { mx -= rx; mz -= rz; }
    const double ml = sqrt(mx * mx + mz * mz);
    if (ml > 1e-6) { mx /= ml; mz /= ml; }

    c->obs_x += (float)(mx * speed * dt);
    c->obs_z += (float)(mz * speed * dt);
    if (mask & UI_LL_ROAM_UP)   c->obs_y += (float)(speed * dt);
    if (mask & UI_LL_ROAM_DOWN) c->obs_y -= (float)(speed * dt);
    if (c->obs_y < 1.5f) c->obs_y = 1.5f;
    c->obs_y = (float)MIN(c->obs_y, 20000.0f);
    // 不钻地：地形就绪时抬到地面以上（与 celestial_ocean 内部的钳制一致，
    // 这里先夹一次是为了让“观察高度”显示与渲染同帧一致）
    if (c->terrain_enable && linglong_ocean_terrain_ready()) {
        double g = linglong_ocean_terrain_height((double)c->obs_x, (double)c->obs_z);
        if ((double)c->obs_y < g + 1.5) c->obs_y = (float)(g + 1.5);
    }
}

// 玲珑仪主天空界面渲染。
// 每帧总是完整重绘（没有“暂停就只画一次”的跳过逻辑）：实时海洋的波场、体积云的
// 飘移都按墙钟推进（get_timestamp_in_ms），与模拟时间是否流动无关——时钟暂停时画面
// 仍在持续变化，跳过重绘会把它们冻住；太阳移动时还需要按帧重建大气 LUT。
// 故本函数无返回值，调用方每帧都会得到一帧全新画面，可安全叠加半透明控件
// （不会出现 alpha 逐帧累积加深）。
void ui_app_linglong_draw_full(Key_Event *key_event, Global_State *global_state) {

    Linglong_Config *llcfg = global_state->linglong_cfg;

    // FPS统计
    static uint64_t fps_last_timestamp = 0;
    static uint32_t fps_frame_count = 0;
    static float fps_display_value = 0.0f;

    fps_frame_count++;
    if (fps_last_timestamp == 0) {
        fps_last_timestamp = global_state->timestamp;
    }
    else if (global_state->timestamp - fps_last_timestamp >= 1000) {
        fps_display_value = fps_frame_count * 1000.0f / (float)(global_state->timestamp - fps_last_timestamp);
        fps_frame_count = 0;
        fps_last_timestamp = global_state->timestamp;
    }

    time_t ts = (time_t)(global_state->timestamp / 1000);

    if (linglong_timemachine_running_state == 1) {
        linglong_timemachine_start_timestamp += (linglong_timemachine_speed * 1000);
        ts = (time_t)(linglong_timemachine_start_timestamp / 1000);
        struct tm *timeinfo = localtime(&ts); // 转换为本地时间

        llcfg->second = timeinfo->tm_sec;
        llcfg->minute = timeinfo->tm_min;
        llcfg->hour = timeinfo->tm_hour;
        llcfg->day = timeinfo->tm_mday;
        llcfg->month = timeinfo->tm_mon + 1;
        llcfg->year = timeinfo->tm_year + 1900;
    }
    else if (linglong_timemachine_running_state == 2) {
        ts = (time_t)(global_state->timestamp / 1000);
        struct tm *timeinfo = localtime(&ts);
        llcfg->second = timeinfo->tm_sec;
        llcfg->minute = timeinfo->tm_min;
        llcfg->hour = timeinfo->tm_hour;
        llcfg->day = timeinfo->tm_mday;
        llcfg->month = timeinfo->tm_mon + 1;
        llcfg->year = timeinfo->tm_year + 1900;
    }


    if (llcfg->enable_imu) {
        llcfg->view_alt  = global_state->pitch;
        llcfg->view_azi  = global_state->yaw + 180.0f;
        llcfg->view_roll = global_state->roll;
    }

    ui_app_linglong_roam_step(key_event, global_state);

    gfx_soft_clear(global_state->llgfx);

    render_sky(global_state->llgfx,
        MIN(global_state->llgfx->width, global_state->llgfx->height) / 2, global_state->llgfx->width / 2, global_state->llgfx->height / 2,
        llcfg->view_alt, llcfg->view_azi, llcfg->view_roll, llcfg->view_f,
        // 2026, 3, 24, 18, 10, 0, 8.0, 119.0, 31.0,
        llcfg->year, llcfg->month, llcfg->day, llcfg->hour, llcfg->minute, llcfg->second, llcfg->timezone, llcfg->longitude, llcfg->latitude,
        llcfg->downsampling_factor,
        llcfg->enable_opt_sym,
        llcfg->enable_opt_lut,
        llcfg->enable_opt_bilinear,
        llcfg->projection,
        llcfg->sky_model,
        llcfg->landscape_index,
        llcfg->enable_equatorial_coord,
        llcfg->enable_horizontal_coord,
        llcfg->enable_star_burst,
        llcfg->enable_star_name,
        llcfg->enable_planet,
        llcfg->enable_ecliptic_circle,
        llcfg->enable_att_indicator,
        llcfg->enable_tracking_sun,
        llcfg->cloud_coverage_level,
        llcfg->cloud_layer_mask,
        llcfg->cloud_precision_level,
        llcfg->ocean_swell_hs_level,
        llcfg->ocean_swell_len_level,
        llcfg->obs_x, llcfg->obs_y, llcfg->obs_z,
        llcfg->terrain_enable, llcfg->terrain_quality, llcfg->terrain_res, llcfg->terrain_seed
    );

    // 观察者位置回读：海洋模块内部做了水平周期回绕与“不钻地”钳制，
    // 把归一化后的值写回配置（否则漫游跨过世界边界时界面数值会越滚越大）。
    {
        double ox = 0.0, oy = 0.0, oz = 0.0;
        linglong_ocean_get_observer(&ox, &oy, &oz);
        llcfg->obs_x = (float)ox;
        llcfg->obs_y = (float)oy;
        llcfg->obs_z = (float)oz;
    }

#if defined(NANO_POD_POCKETTERM35)
    // PocketTerm35 的 HDMI 屏经 32bpp framebuffer 输出 8bit/通道时无需 565 抖动
    //（gfx_dithering 会预量化到 565 色阶，反而成为色深瓶颈）；fb 为 16bpp 时保留
    if (display_hal_get_color_depth() <= 16) gfx_dithering(global_state->llgfx);
#else
    gfx_dithering(global_state->llgfx);
#endif
    // gfx_gamma(global_state->llgfx, 1.3f);

    // 显示FPS
    wchar_t fps_str[16];
    swprintf(fps_str, 16, L"FPS=%.1f", fps_display_value);
    gfx_font_draw_text(global_state->llgfx, GFX_FONT_ALPHA_12, fps_str, 1, 0, 0, 255, 0, 1);
}

void ui_app_linglong_draw_lite(
    Key_Event *key_event, Global_State *global_state,
    int32_t x, int32_t y,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double longitude, double latitude, double timezone
) {
    uint8_t BG_R = 255, BG_G = 255, BG_B = 255;
    uint8_t COORD_R = 222, COORD_G = 222, COORD_B = 222;
    uint8_t NSWE_R = 255, NSWE_G = 0, NSWE_B = 0;
    uint8_t DATETIME_R = 0, DATETIME_G = 0, DATETIME_B = 255;
    uint8_t TEXT_R = 0, TEXT_G = 0, TEXT_B = 0;
    uint8_t SUN_R = 255, SUN_G = 0, SUN_B = 0;
    uint8_t MOON_R = 255, MOON_G = 0, MOON_B = 255;

    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        // 同初始化
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        BG_R = 0, BG_G = 0, BG_B = 0;
        COORD_R = 66, COORD_G = 66, COORD_B = 66;
        NSWE_R = 255, NSWE_G = 0, NSWE_B = 0;
        DATETIME_R = 0, DATETIME_G = 255, DATETIME_B = 255;
        TEXT_R = 255, TEXT_G = 255, TEXT_B = 255;
        SUN_R = 255, SUN_G = 255, SUN_B = 0;
        MOON_R = 222, MOON_G = 222, MOON_B = 0;
    }

    gfx_draw_rectangle(global_state->gfx, x, y, 128, 64, BG_R, BG_G, BG_B, 1);

    gfx_draw_circle(global_state->gfx, x+64, y+32, 30,         COORD_R, COORD_G, COORD_B, 1);
    gfx_draw_circle(global_state->gfx, x+64, y+32, 20,         COORD_R, COORD_G, COORD_B, 1);
    gfx_draw_circle(global_state->gfx, x+64, y+32, 10,         COORD_R, COORD_G, COORD_B, 1);
    gfx_draw_line(global_state->gfx,   x+32, y+32, x+96, y+32, COORD_R, COORD_G, COORD_B, 1);
    gfx_draw_line(global_state->gfx,   x+64, y+0, x+64, y+64,  COORD_R, COORD_G, COORD_B, 1);

    gfx_draw_rectangle(global_state->gfx, x+62-1, y+0,    5+2, 5+1, BG_R, BG_G, BG_B, 1); // N背景
    gfx_draw_rectangle(global_state->gfx, x+63-1, y+59-1, 3+2, 5+1, BG_R, BG_G, BG_B, 1); // S背景
    gfx_draw_rectangle(global_state->gfx, x+32-1, y+30-1, 5+2, 5+2, BG_R, BG_G, BG_B, 1); // W背景
    gfx_draw_rectangle(global_state->gfx, x+93-1, y+30-1, 3+2, 5+2, BG_R, BG_G, BG_B, 1); // E背景

    // 方位文字和周围的边框
    gfx_draw_textline_mini(global_state->gfx, L"N", x+62, y+0,  NSWE_R, NSWE_G, NSWE_B, 1); gfx_draw_point(global_state->gfx, x+61, y+2,  BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+67, y+2,  BG_R, BG_G, BG_B, 1);  gfx_draw_point(global_state->gfx, x+64, y+5, BG_R, BG_G, BG_B, 1);
    gfx_draw_textline_mini(global_state->gfx, L"S", x+63, y+59, NSWE_R, NSWE_G, NSWE_B, 1); gfx_draw_point(global_state->gfx, x+62, y+62, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+66, y+62, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+64, y+58, BG_R, BG_G, BG_B, 1);
    gfx_draw_textline_mini(global_state->gfx, L"W", x+32, y+30, NSWE_R, NSWE_G, NSWE_B, 1); gfx_draw_point(global_state->gfx, x+34, y+29, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+37, y+32, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+34, y+35, BG_R, BG_G, BG_B, 1);
    gfx_draw_textline_mini(global_state->gfx, L"E", x+93, y+30, NSWE_R, NSWE_G, NSWE_B, 1); gfx_draw_point(global_state->gfx, x+92, y+32, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+96, y+32, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+94, y+29, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+94, y+35, BG_R, BG_G, BG_B, 1);

    gfx_draw_line(global_state->gfx, x+0, y+43, x+30, y+43, COORD_R, COORD_G, COORD_B, 1);

    wchar_t timestr[30];
    swprintf(timestr, 30, L"%04d-%02d-%02d\n%02d:%02d:%02d", year, month, day, hour, minute, second);
    gfx_draw_textline_mini(global_state->gfx, timestr, x+0, y+0, DATETIME_R, DATETIME_G, DATETIME_B, 1);

    double altitude_moon = 0.0;
    double azimuth_moon = 0.0;

    where_is_the_moon(year, month, day, hour, minute, second, timezone, longitude, latitude, &azimuth_moon, &altitude_moon);
    double i_deg = moon_phase(year, month, day, hour, minute, second, timezone);
    double moon_k = (1.0 + cos(i_deg / 180.0 * M_PI)) / 2.0;

    wchar_t coordstr_moon[30];
    swprintf(coordstr_moon, 30, L"MOON\nP:%d%%\nA:%.1f\nE:%.1f", (int32_t)(moon_k * 100.0), azimuth_moon, altitude_moon);
    gfx_draw_textline_mini(global_state->gfx, coordstr_moon, x+0, y+18, TEXT_R, TEXT_G, TEXT_B, 1);

    double x_moon = 64 + (90.0 - altitude_moon) * 32.0 / 90.0 * sin(azimuth_moon / 180.0 * M_PI);
    double y_moon = 32 - (90.0 - altitude_moon) * 32.0 / 90.0 * cos(azimuth_moon / 180.0 * M_PI);

    if (x_moon >= 32 && x_moon <= 96 && y_moon >= 0 && y_moon <= 64) {
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 1, y + (int)y_moon - 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 1, y + (int)y_moon - 0, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 1, y + (int)y_moon + 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 0, y + (int)y_moon - 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 0, y + (int)y_moon - 0, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 0, y + (int)y_moon + 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon + 1, y + (int)y_moon - 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon + 1, y + (int)y_moon - 0, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon + 1, y + (int)y_moon + 1, MOON_R, MOON_G, MOON_B, 1);
    }

    double altitude_sun = 0.0;
    double azimuth_sun = 0.0;

    where_is_the_sun(year, month, day, hour, minute, second, +8.0, longitude, latitude, &azimuth_sun, &altitude_sun);

    wchar_t coordstr_sun[30];
    swprintf(coordstr_sun, 30, L"SUN\nA:%.1f\nE:%.1f", azimuth_sun, altitude_sun);
    gfx_draw_textline_mini(global_state->gfx, coordstr_sun, x+0, y+46, TEXT_R, TEXT_G, TEXT_B, 1);

    double x_sun = 64 + (90.0 - altitude_sun) * 32.0 / 90.0 * sin(azimuth_sun / 180.0 * M_PI);
    double y_sun = 32 - (90.0 - altitude_sun) * 32.0 / 90.0 * cos(azimuth_sun / 180.0 * M_PI);

    if (x_sun >= 32 && x_sun <= 96 && y_sun >= 0 && y_sun <= 64) {
        gfx_draw_circle(global_state->gfx, x+(int)x_sun, y+(int)y_sun, 2, SUN_R, SUN_G, SUN_B, 1);
    }


    // 二分搜索日出日落时间
    if (linglong_first_call_timestamp == 0 || linglong_last_day != day) { // 只在首次调用和当天日期变化时计算
        linglong_first_call_timestamp = global_state->timestamp;
        linglong_last_day = day;

        int32_t sunrise_min = find_sunrise(year, month, day, timezone, longitude, latitude);
        if (sunrise_min != -1) {
            linglong_sunrise_time[0] = sunrise_min / 60;
            linglong_sunrise_time[1] = sunrise_min % 60;
        }
        int32_t sunset_min = find_sunset(year, month, day, timezone, longitude, latitude);
        if (sunset_min != -1) {
            linglong_sunset_time[0] = sunset_min / 60;
            linglong_sunset_time[1] = sunset_min % 60;
        }
    }
    wchar_t risefall_time[60];
    swprintf(risefall_time, 60, L"R:%02d:%02d\nS:%02d:%02d", linglong_sunrise_time[0], linglong_sunrise_time[1], linglong_sunset_time[0], linglong_sunset_time[1]);
    gfx_draw_textline_mini(global_state->gfx, risefall_time, x+98, y+0, TEXT_R, TEXT_G, TEXT_B, 1);

    gfx_draw_textline_mini(global_state->gfx, L"    BD4SUR\n 2011-2026", x+86, y+53, TEXT_R, TEXT_G, TEXT_B, 1);
}

void ui_app_linglong_render_frame(Key_Event *key_event, Global_State *global_state) {
    // 天空每帧都完整重绘；设置界面与主控制面板都是叠加在其上的半透明控件，
    // 因此不会出现 alpha 逐帧累积加深。
    ui_app_linglong_draw_full(key_event, global_state);

    if (linglong_state == LL_STATE_SETTING || linglong_state == LL_STATE_SETTING_CALLBACK) {
        ui_app_linglong_setting_draw(key_event, global_state);
    }
    gfx_font_draw_text(global_state->llgfx, GFX_FONT_ALPHA_12, (wchar_t *)L"玲珑天象仪 V" NANO_VERSION,
        1, global_state->llgfx->height - gfx_font_line_height(GFX_FONT_ALPHA_12), 255, 255, 255, 200);

    wchar_t timestr[30];
    swprintf(timestr, 30, L"%ls %04d-%02d-%02d %02d:%02d:%02d",
        (linglong_timemachine_running_state == 0) ? L"  " :
            (((linglong_timemachine_running_state == 1) && (linglong_timemachine_speed > 0)) ? L">>" :
            (((linglong_timemachine_running_state == 1) && (linglong_timemachine_speed < 0)) ? L"<<" : L" >")),
        global_state->linglong_cfg->year, global_state->linglong_cfg->month, global_state->linglong_cfg->day, global_state->linglong_cfg->hour, global_state->linglong_cfg->minute, global_state->linglong_cfg->second);
    int32_t timestr_x = global_state->llgfx->width - gfx_font_measure_text(GFX_FONT_ALPHA_12, timestr) - 1;
    if (timestr_x < 0) timestr_x = 0;
    gfx_font_draw_text(global_state->llgfx, GFX_FONT_ALPHA_12, timestr,
        timestr_x, global_state->llgfx->height - gfx_font_line_height(GFX_FONT_ALPHA_12), 255, 255, 255, 1);

    // 主天空界面：叠加显式触屏控件（顶部按钮栏 + 可选的时间/视角面板）。
    // 设置界面（LL_STATE_SETTING*）不绘制，继续沿用原有十六宫格软键布局。
    if (linglong_state == LL_STATE_SKY) {
        UiLinglongCtrlState ctrl_state;
        ctrl_state.running_state = linglong_timemachine_running_state;
        ctrl_state.speed = linglong_timemachine_speed;
        ctrl_state.projection = global_state->linglong_cfg->projection;
        ctrl_state.equatorial_coord = global_state->linglong_cfg->enable_equatorial_coord;
        ctrl_state.horizontal_coord = global_state->linglong_cfg->enable_horizontal_coord;
        ctrl_state.sky_model = global_state->linglong_cfg->sky_model;
        ctrl_state.landscape_index = global_state->linglong_cfg->landscape_index;
        ctrl_state.roam_enable = global_state->linglong_cfg->roam_enable;
        ctrl_state.roam_speed_level = global_state->linglong_cfg->roam_speed_level;
        ui_linglong_ctrl_draw(global_state->llgfx, &ctrl_state);
    }

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    convert_rgb888_to_rgb565_double(global_state->gfx, global_state->llgfx->frame_buffer_rgb888, global_state->llgfx->width, global_state->llgfx->height);
    gfx_refresh(global_state->gfx);
#else
    // convert_rgb888_to_rgb565_double(global_state->gfx, global_state->llgfx->frame_buffer_rgb888, global_state->llgfx->width, global_state->llgfx->height);
    gfx_refresh(global_state->llgfx);
#endif

}


void ui_app_linglong_toggle_timemachine(Key_Event *key_event, Global_State *global_state) {
    if (linglong_timemachine_running_state == 0) {
        linglong_timemachine_running_state = 1;
    }
    else {
        linglong_timemachine_running_state = 0;
    }
    if (linglong_timemachine_start_timestamp == 0) {
        linglong_timemachine_start_timestamp = global_state->timestamp;
    }
}

void ui_app_linglong_set_timemachine_speed(Key_Event *key_event, Global_State *global_state, int32_t speed) {
    linglong_timemachine_speed = speed;
    switch (linglong_timemachine_running_state) {
        case 0: linglong_timemachine_running_state = 1; break;
        case 1: linglong_timemachine_running_state = 0; break;
        case 2: linglong_timemachine_running_state = 1; break;
        default: linglong_timemachine_running_state = 0; break;
    }
    if (linglong_timemachine_start_timestamp == 0) {
        linglong_timemachine_start_timestamp = global_state->timestamp;
    }
}

void ui_app_linglong_set_realtime(Key_Event *key_event, Global_State *global_state) {
    Linglong_Config *llcfg = global_state->linglong_cfg;
    time_t ts = (time_t)(global_state->timestamp / 1000);
    struct tm *timeinfo = localtime(&ts); // 转换为本地时间
    llcfg->second = timeinfo->tm_sec;
    llcfg->minute = timeinfo->tm_min;
    llcfg->hour = timeinfo->tm_hour;
    llcfg->day = timeinfo->tm_mday;
    llcfg->month = timeinfo->tm_mon + 1;
    llcfg->year = timeinfo->tm_year + 1900;

    if (linglong_timemachine_running_state == 0) {
        linglong_timemachine_running_state = 2;
    }
    else {
        linglong_timemachine_running_state = 0;
    }
    if (linglong_timemachine_start_timestamp == 0) {
        linglong_timemachine_start_timestamp = global_state->timestamp;
    }
}

// 显式触屏控件：恢复当前时间（始终回到实时，不做“再按一次暂停”的开关语义）
static void ui_app_linglong_control_realtime(Global_State *global_state) {
    Linglong_Config *llcfg = global_state->linglong_cfg;
    time_t ts = (time_t)(global_state->timestamp / 1000);
    struct tm *timeinfo = localtime(&ts);
    llcfg->second = timeinfo->tm_sec;
    llcfg->minute = timeinfo->tm_min;
    llcfg->hour   = timeinfo->tm_hour;
    llcfg->day    = timeinfo->tm_mday;
    llcfg->month  = timeinfo->tm_mon + 1;
    llcfg->year   = timeinfo->tm_year + 1900;

    linglong_timemachine_running_state = 2;
    linglong_paused_resume_state = 2;
}

// 显式触屏控件：运行/暂停（暂停记住暂停前的模式，恢复时回到该模式）
static void ui_app_linglong_control_runpause(Global_State *global_state) {
    if (linglong_timemachine_running_state == 0) {
        linglong_timemachine_running_state = linglong_paused_resume_state;
        // 兜底：只可能恢复到“时光机”或“实时”
        if (linglong_timemachine_running_state != 1) {
            linglong_timemachine_running_state = 2;
        }
    }
    else {
        linglong_paused_resume_state = linglong_timemachine_running_state;
        linglong_timemachine_running_state = 0;
    }
}

// ===============================================================================
// 视角按钮的“按住反复触发”
// ===============================================================================
// 视角九宫格的 8 枚方向/缩放按钮（正中“归中”除外）支持按住反复触发，交互与漫游移动 pad
// 一致：按住不放即连续转动/缩放视角。触发节奏取键盘自动重复的两段式——
//   · 短按（< 初始延时）：只由松手沿的单次点击完成（与改动前完全一致，拖动离手不触发）；
//   · 长按：超过初始延时后按固定间隔重复同一步进（5° / 0.1f，约 70°/s），松手即停。
// 状态按“当前按住的按钮”跟踪：按下点移到别的按钮、面板隐藏或松手都会重新计时/停掉，
// 因而换按钮不会有上一枚按钮的重复惯性（状态无需显式复位：每帧轮询电平自行收敛）。
#define LL_VIEW_REPEAT_DELAY_MS     (320)
#define LL_VIEW_REPEAT_INTERVAL_MS  (70)

static int32_t  s_view_repeat_action = UI_LL_CTRL_NONE;
static uint64_t s_view_repeat_next_ms = 0;

// 每帧调用一次；返回本帧应重复触发的视角动作（UI_LL_CTRL_VIEW_n），否则 NONE
static int32_t ui_app_linglong_view_repeat(int32_t is_touching, int32_t x, int32_t y) {
    int32_t held = ui_linglong_ctrl_view_held(is_touching, x, y);
    if (held == UI_LL_CTRL_NONE) {
        s_view_repeat_action = UI_LL_CTRL_NONE; // 松手/移出按钮/面板隐藏：停掉重复
        return UI_LL_CTRL_NONE;
    }
    const uint64_t now = get_timestamp_in_ms();
    if (held != s_view_repeat_action) {
        s_view_repeat_action = held;                          // 新的按住：先等初始延时
        s_view_repeat_next_ms = now + LL_VIEW_REPEAT_DELAY_MS;
        return UI_LL_CTRL_NONE;
    }
    if (now < s_view_repeat_next_ms) return UI_LL_CTRL_NONE;
    s_view_repeat_next_ms = now + LL_VIEW_REPEAT_INTERVAL_MS;
    return held;
}

void ui_app_linglong_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 获取机器姿态（欧拉角）
#ifdef IMU_ENABLED

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)

#else
    if (global_state->linglong_cfg->enable_imu) {
        imu_read_angle(&(global_state->pitch), &(global_state->roll), &(global_state->yaw));
        printf("俯仰=%-10.2f    滚转=%-10.2f    航向=%-10.2f\n", global_state->pitch, global_state->roll, global_state->yaw);
    }
#endif

#endif

    // 主天空界面显式触屏控件（已无启动画面停留期，进入即刻接管触屏）。
    // 设置界面另有 5x4 显式按钮，见下方独立分支。
    if (linglong_state == LL_STATE_SKY) {
        // 触屏拖动变换视角：仅在控制面板隐藏时启用（控制模块内部判断），
        // 按下点落在顶部按钮上时不启动拖动。像素增量按实际画布尺寸归一化：
        // 横/纵向拖满整屏分别对应 180° 方位/俯仰变化，不写死屏幕宽高。
        // 方向为“抓取天空”：画面随手指同向移动——右拖天空右移（视角左转）、
        // 下拖天空下移（视角抬头）；即与十六宫格数字键的“相机转向”方向相反。
        int32_t drag_dx = 0, drag_dy = 0;
        if (ui_linglong_ctrl_poll_drag(
                key_event->touch_edge, key_event->is_touching,
                key_event->touch_x, key_event->touch_y,
                key_event->touch_down_x, key_event->touch_down_y,
                &drag_dx, &drag_dy)) {
            Linglong_Config *llcfg = global_state->linglong_cfg;
            int32_t canvas_w = (int32_t)global_state->llgfx->width;
            int32_t canvas_h = (int32_t)global_state->llgfx->height;
            if (canvas_w > 0 && canvas_h > 0) {
                llcfg->enable_imu = 0; // 手动拖动，关闭IMU
                llcfg->view_azi -= (float)drag_dx * 180.0f / (float)canvas_w;
                while (llcfg->view_azi >= 360.0f) llcfg->view_azi -= 360.0f;
                while (llcfg->view_azi < 0.0f) llcfg->view_azi += 360.0f;
                llcfg->view_alt += (float)drag_dy * 180.0f / (float)canvas_h;
                if (llcfg->view_alt > 90.0f) llcfg->view_alt = 90.0f;
                if (llcfg->view_alt < -90.0f) llcfg->view_alt = -90.0f;
            }
        }

        int32_t action = ui_linglong_ctrl_handle_touch(
            key_event->touch_edge, key_event->touch_x, key_event->touch_y,
            key_event->touch_down_x, key_event->touch_down_y);

        // 视角方向按钮的按住重复：逐帧轮询触点（与漫游 pad 同一范式，与触屏边沿无关）
        int32_t repeat_action = ui_app_linglong_view_repeat(
            key_event->is_touching, key_event->touch_x, key_event->touch_y);

        // 控件显隐/按住高亮变化无需特殊处理：玲珑仪每帧都完整重绘（暂停也不例外），
        // 控件状态每帧都会被重画在新画面上；这里只把脏标记消费掉，避免它一直挂着。
        (void)ui_linglong_ctrl_take_dirty();

        if (action != UI_LL_CTRL_NONE) {
            int32_t synth_key = NANO_KEY_IDLE;

            switch (action) {
                case UI_LL_CTRL_SETTINGS:
                    // 控制面板右上角“设置”：打开 5x4 显式触屏设置界面
                    global_state->is_ctrl_enabled = 1;
                    linglong_state = LL_STATE_SETTING;
                    ui_linglong_ctrl_reset();
                    ui_linglong_ctrl_settings_reset();
                    s_linglong_touch_owned = 1; // 设置界面也用显式触屏控件
                    return;

                case UI_LL_CTRL_EXIT:
                    // 等价于十六宫格 A 键：退出玲珑仪
                    global_state->is_ctrl_enabled = 0;
                    global_state->STATE = STATE_MAIN_MENU;
                    s_linglong_touch_owned = 0;
                    return;

                case UI_LL_CTRL_REWIND:
                    // 沿用十六宫格“*”键行为：设置快退速度并切换启停（再点一次即暂停）
                    ui_app_linglong_set_timemachine_speed(key_event, global_state, -120);
                    linglong_paused_resume_state = 1;
                    break;
                case UI_LL_CTRL_REALTIME:
                    ui_app_linglong_control_realtime(global_state);
                    break;
                case UI_LL_CTRL_RUNPAUSE:
                    ui_app_linglong_control_runpause(global_state);
                    break;
                case UI_LL_CTRL_FORWARD:
                    // 沿用十六宫格“#”键行为：设置快进速度并切换启停（再点一次即暂停）
                    ui_app_linglong_set_timemachine_speed(key_event, global_state, 120);
                    linglong_paused_resume_state = 1;
                    break;

                case UI_LL_CTRL_PROJECTION:
                    // 投影算法：鱼眼 ↔ 透视
                    global_state->linglong_cfg->projection =
                        (global_state->linglong_cfg->projection == 0) ? 1 : 0;
                    break;
                case UI_LL_CTRL_EQUATORIAL:
                    // 赤道坐标：关 → 开 → 关
                    global_state->linglong_cfg->enable_equatorial_coord =
                        (global_state->linglong_cfg->enable_equatorial_coord + 1) % 2;
                    break;
                case UI_LL_CTRL_HORIZONTAL:
                    // 地平坐标：关 → 方位角 → 坐标圈 → 关
                    global_state->linglong_cfg->enable_horizontal_coord =
                        (global_state->linglong_cfg->enable_horizontal_coord + 1) % 3;
                    break;
                case UI_LL_CTRL_SKY_MODEL:
                    // 大气散射：关/二次散射/体积云/海洋天空 循环（简化模型、一次散射已屏蔽）
                    global_state->linglong_cfg->sky_model =
                        ui_app_linglong_next_sky_model(global_state->linglong_cfg->sky_model);
                    break;
                case UI_LL_CTRL_LANDSCAPE:
                    // 地景：0..3 循环
                    global_state->linglong_cfg->landscape_index =
                        (global_state->linglong_cfg->landscape_index + 1) % 4;
                    break;
                case UI_LL_CTRL_ROAM:
                    // 漫游模式开关（顶部第三枚按钮）：开启后底部行换成 6 键移动 pad
                    global_state->linglong_cfg->roam_enable =
                        (global_state->linglong_cfg->roam_enable != 0) ? 0 : 1;
                    if (global_state->linglong_cfg->roam_enable) {
                        s_roam_last_ms = 0;   // 避免用上一段漫游的 dt
                    }
                    break;
                case UI_LL_CTRL_MOVE_FWD:
                case UI_LL_CTRL_MOVE_BACK:
                case UI_LL_CTRL_MOVE_LEFT:
                case UI_LL_CTRL_MOVE_RIGHT:
                case UI_LL_CTRL_MOVE_UP:
                case UI_LL_CTRL_MOVE_DOWN:
                    // 移动 pad 是“按住连续移动”（ui_linglong_ctrl_roam_held 每帧轮询），
                    // 松手沿的点击不产生额外动作
                    break;

                default:
                    // 视角九宫格：合成对应的 NANO_KEY_1~9 短按事件，
                    // 复用下方既有硬键处理逻辑（功能与十六宫格软键 1~9 一一对应）。
                    // 8 枚方向/缩放按钮另有“按住反复触发”（见 ui_app_linglong_view_repeat），
                    // 短按仍走这里的一次触发。
                    if (action >= UI_LL_CTRL_VIEW_1 && action <= UI_LL_CTRL_VIEW_9) {
                        global_state->is_ctrl_enabled = 0;
                        synth_key = NANO_KEY_1 + (action - UI_LL_CTRL_VIEW_1);
                    }
                    break;
            }

            if (synth_key != NANO_KEY_IDLE) {
                key_event->key_code = (uint8_t)synth_key;
                key_event->key_edge = -1; // 合成短按（松手沿已过，直接视为一次完整点击）
                key_event->is_soft_key = 1;
            }
            else {
                // 时间控制已直接处理，吞掉本次事件，避免与同时按下的硬键重复响应
                key_event->key_code = NANO_KEY_IDLE;
                key_event->key_edge = 0;
            }
        }
        else if (repeat_action != UI_LL_CTRL_NONE && key_event->key_edge == 0) {
            // 按住反复触发：合成同键码的长按重复事件（key_edge=-2，与实体键长按同一语义），
            // 复用下方既有 NANO_KEY_1..9 处理链，保证“点按”与“按住”两条路径行为一致。
            // 只在“无沿事件”的帧注入（key_edge==0：空闲或实体键按住中）——此时不会丢掉
            // 任何上升/下降沿，而实体键按住（如漫游中按着 W）也不会阻断视角重复。
            global_state->is_ctrl_enabled = 0;
            key_event->key_code = (uint8_t)(NANO_KEY_1 + (repeat_action - UI_LL_CTRL_VIEW_1));
            key_event->key_edge = -2;
            key_event->is_soft_key = 1;
        }
    }

    // 设置界面显式触屏控件：5x4 按钮，替代旧 4x4 宫格软键。
    // 命中功能格后合成对应 NANO_KEY_* 短按事件，复用下方既有 Ctrl+键 处理链。
    if (linglong_state == LL_STATE_SETTING || linglong_state == LL_STATE_SETTING_CALLBACK) {
        int32_t cell = ui_linglong_ctrl_settings_handle_touch(
            key_event->touch_edge, key_event->touch_x, key_event->touch_y,
            key_event->touch_down_x, key_event->touch_down_y);
        if (cell >= 0 && cell <= 2) {
            // 云参数：直接循环档位（硬键没有空位可映射，与海洋参数按钮同为触屏专用）
            if (cell == 0) {
                ui_app_linglong_cloud_next_coverage(global_state);
            }
            else if (cell == 1) {
                ui_app_linglong_cloud_next_layer(global_state);
            }
            else {
                ui_app_linglong_cloud_next_precision(global_state);
            }
            linglong_state = LL_STATE_SETTING_CALLBACK;   // 重绘设置界面（现值已变）
        }
        else if (cell >= 11 && cell <= 13) {
            // 海洋参数：直接循环档位（硬键没有空位可映射，与云参数按钮同为触屏专用）
            Linglong_Config *oc = global_state->linglong_cfg;
            int32_t n = 0;
            if (cell == 11) {
                // 观察高度：档位表是“快捷跳转”（真实高度在漫游中是连续量）
                const Linglong_Ocean_Option *ho = linglong_ocean_height_options(&n);
                if (n > 0) {
                    oc->ocean_height_level = (oc->ocean_height_level + 1) % n;
                    if (ho != NULL) oc->obs_y = ho[oc->ocean_height_level].value;
                }
            }
            else if (cell == 12) {
                linglong_ocean_swell_hs_options(&n);
                oc->ocean_swell_hs_level = (oc->ocean_swell_hs_level + 1) % (n > 0 ? n : 1);
            }
            else {
                linglong_ocean_swell_len_options(&n);
                oc->ocean_swell_len_level = (oc->ocean_swell_len_level + 1) % (n > 0 ? n : 1);
            }
            linglong_state = LL_STATE_SETTING_CALLBACK;   // 重绘设置界面（现值已变）
        }
        else if (cell == 10) {
            // 岛屿地形开关（landscape_index==3 时生效）
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->terrain_enable = (oc->terrain_enable != 0) ? 0 : 1;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell == 14) {
            // 地形精度：省电 → 标准 → 精细 → 省电
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->terrain_quality = (oc->terrain_quality + 1) % 3;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell == 15) {
            // 换一座岛：种子 +1 并立即作废重建（渐进生成，界面不冻结）
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->terrain_seed = (oc->terrain_seed == 0) ? 1u : oc->terrain_seed + 1u;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell == 16) {
            // 高程图分辨率：512（50m 纹素）↔ 1024（25m 纹素）
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->terrain_res = (oc->terrain_res >= 1024) ? 512 : 1024;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell == 17) {
            // 漫游速度：10 / 100 / 1000 m/s
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->roam_speed_level = (oc->roam_speed_level + 1) % 3;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell == 19) {
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->roam_enable = (oc->roam_enable != 0) ? 0 : 1;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell == 18) {
            // 观察位置：只读显示
        }
        else if (cell >= 0) {
            int32_t synth_key = ui_app_linglong_setting_cell_key(cell);
            if (synth_key != NANO_KEY_IDLE) {
                global_state->is_ctrl_enabled = 1;
                key_event->key_code = (uint8_t)synth_key;
                key_event->key_edge = -1;
                key_event->is_soft_key = 1;
            }
        }
    }

    // 漫游：方向键 / W A S D / Q E 按住连续移动（漫游开启时方向键不再兼作时光机）
    if (global_state->linglong_cfg->roam_enable &&
        (linglong_state == LL_STATE_SKY || linglong_state == LL_STATE_SETTING ||
         linglong_state == LL_STATE_SETTING_CALLBACK)) {
        switch (key_event->key_code) {
            case NANO_KEY_up:    case NANO_KEY_W: case NANO_KEY_w: ui_app_linglong_roam_key_hold(UI_LL_ROAM_FWD);   break;
            case NANO_KEY_down:  case NANO_KEY_S: case NANO_KEY_s: ui_app_linglong_roam_key_hold(UI_LL_ROAM_BACK);  break;
            case NANO_KEY_left:  case NANO_KEY_A: case NANO_KEY_a: ui_app_linglong_roam_key_hold(UI_LL_ROAM_LEFT);  break;
            case NANO_KEY_right: case NANO_KEY_D: case NANO_KEY_d: ui_app_linglong_roam_key_hold(UI_LL_ROAM_RIGHT); break;
            case NANO_KEY_E:     case NANO_KEY_e: case NANO_KEY_pgup: ui_app_linglong_roam_key_hold(UI_LL_ROAM_UP);   break;
            case NANO_KEY_Q:     case NANO_KEY_q: case NANO_KEY_pgdn: ui_app_linglong_roam_key_hold(UI_LL_ROAM_DOWN); break;
            default: break;
        }
    }

    int32_t is_setting_refresh = 0;

    // 按1键向左偏航（yaw--），或者Ctrl时切换投影算法
    if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_1) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_azi -= 5.0f;
            if (global_state->linglong_cfg->view_azi <= 0.0f) {
                global_state->linglong_cfg->view_azi = 360.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            if (global_state->linglong_cfg->projection == 0) {
                global_state->linglong_cfg->projection = 1;
            }
            else {
                global_state->linglong_cfg->projection = 0;
            }
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按2键推杆低头（pitch--），或者Ctrl时切换赤道坐标圈
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_2) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_alt -= 5.0f;
            if (global_state->linglong_cfg->view_alt <= -90.0f) {
                global_state->linglong_cfg->view_alt = -90.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_equatorial_coord ++;
            global_state->linglong_cfg->enable_equatorial_coord = global_state->linglong_cfg->enable_equatorial_coord % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按3键向右偏航（yaw++），或者Ctrl时切换地平坐标
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_3) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_azi += 5.0f;
            if (global_state->linglong_cfg->view_azi >= 360.0f) {
                global_state->linglong_cfg->view_azi = 0.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_horizontal_coord++;
            global_state->linglong_cfg->enable_horizontal_coord = global_state->linglong_cfg->enable_horizontal_coord % 3;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按4键向左坡度（roll--），或者Ctrl时切换黄道
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_4) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_roll -= 5.0f;
            if (global_state->linglong_cfg->view_roll <= -90.0f) {
                global_state->linglong_cfg->view_roll = -90.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_ecliptic_circle++;
            global_state->linglong_cfg->enable_ecliptic_circle = global_state->linglong_cfg->enable_ecliptic_circle % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按5键归中，或切换陀螺仪状态，或者Ctrl时切换天体名称
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_5) {
        if (global_state->is_ctrl_enabled == 0) {
            // 如果IMU已关闭，则开启
            if (global_state->linglong_cfg->enable_imu == 0) {
                global_state->linglong_cfg->enable_imu = 1;
            }
            // 如果IMU已开启，则关闭并归中
            else {
                global_state->linglong_cfg->enable_imu = 0;
                global_state->linglong_cfg->view_alt = 90.0f;
                global_state->linglong_cfg->view_azi = 180.0f;
                global_state->linglong_cfg->view_f = 1.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_star_name++;
            global_state->linglong_cfg->enable_star_name = global_state->linglong_cfg->enable_star_name % 4;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按6键向右坡度（roll++），或者Ctrl时切换姿态指示
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_6) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_roll += 5.0f;
            if (global_state->linglong_cfg->view_roll >= 90.0f) {
                global_state->linglong_cfg->view_roll = 90.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_att_indicator++;
            global_state->linglong_cfg->enable_att_indicator = global_state->linglong_cfg->enable_att_indicator % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按7键拉远，或者Ctrl时切换大气散射模型
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_7) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->view_f -= 0.1f;
            if (global_state->linglong_cfg->view_f <= 0.1f) global_state->linglong_cfg->view_f = 0.1f;
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->sky_model =
                ui_app_linglong_next_sky_model(global_state->linglong_cfg->sky_model);
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按8键拉杆抬头（pitch++），或者Ctrl时切换地景
    if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_8) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_alt += 5.0f;
            if (global_state->linglong_cfg->view_alt >= 90.0f) {
                global_state->linglong_cfg->view_alt = 90.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->landscape_index++;
            global_state->linglong_cfg->landscape_index = global_state->linglong_cfg->landscape_index % 4;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按9键推近，或者Ctrl时切换平滑滤波
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_9) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->view_f += 0.1f;
            if (global_state->linglong_cfg->view_f >= 5.0f) global_state->linglong_cfg->view_f = 5.0f;
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_opt_bilinear++;
            global_state->linglong_cfg->enable_opt_bilinear = global_state->linglong_cfg->enable_opt_bilinear % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按A键返回主菜单
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->is_ctrl_enabled = 0;
        global_state->STATE = STATE_MAIN_MENU;
    }
    // 按B键+Ctrl校准IMU
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_shift) {
        if (global_state->is_ctrl_enabled == 0) {
            // TODO
        }
        else {
            // global_state->is_ctrl_enabled = 0;
#ifdef IMU_ENABLED
            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L" \n \n    正在校准IMU...", 0, 0);
            ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
            imu_calib();
            sleep_in_ms(500);
            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L" \n \n    校准完成", 0, 0);
            ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
#endif
        }
    }
    // 按C键切换Ctrl
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_ctrl) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->is_ctrl_enabled = 1;
            linglong_state = LL_STATE_SETTING;
            ui_linglong_ctrl_settings_reset();
        }
        else {
            // 退出设置界面 → 回到控制面板（时间/视角/设置按钮同显），而不是空天空：
            // 设置界面自身没有顶部按钮栏，返回后落在“能继续操作”的地方才顺手。
            global_state->is_ctrl_enabled = 0;
            linglong_state = LL_STATE_SKY;
            ui_linglong_ctrl_show_controls();
        }
    }
    // 按*键时光机向前（过去）（反复按运行/暂停）；漫游模式下方向键用于移动
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_left &&
             global_state->linglong_cfg->roam_enable == 0) {
        ui_app_linglong_set_timemachine_speed(key_event, global_state, -120);
    }
    // 按0键回到实时（反复按运行/暂停），或者Ctrl时切换跟踪太阳
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_0) {
        if (global_state->is_ctrl_enabled == 0) {
            ui_app_linglong_set_realtime(key_event, global_state);
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_tracking_sun++;
            global_state->linglong_cfg->enable_tracking_sun = global_state->linglong_cfg->enable_tracking_sun % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按#键时光机向后（未来）（反复按运行/暂停）；漫游模式下方向键用于移动
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_right &&
             global_state->linglong_cfg->roam_enable == 0) {
        ui_app_linglong_set_timemachine_speed(key_event, global_state, 120);
    }

    // 主天空界面与设置界面都由显式触屏控件接管，抑制旧 4x4 宫格软键；实体硬键兜底。
    s_linglong_touch_owned = (linglong_state == LL_STATE_SKY ||
                              linglong_state == LL_STATE_SETTING ||
                              linglong_state == LL_STATE_SETTING_CALLBACK);
    if (linglong_state != LL_STATE_SKY) {
        // 设置界面不显示主界面控件；无论经控制面板还是硬 C 键进入，返回主界面时
        // 时间/视角面板都从隐藏状态开始，避免旧面板状态残留。
        ui_linglong_ctrl_reset();
    }
    if (linglong_state != LL_STATE_SETTING && linglong_state != LL_STATE_SETTING_CALLBACK) {
        ui_linglong_ctrl_settings_reset();
    }
}




// ===============================================================================
// 设置菜单
// ===============================================================================

static int32_t year_edit = 0;
static int32_t month_edit = 0;
static int32_t day_edit = 0;
static int32_t hour_edit = 0;
static int32_t minute_edit = 0;
static float timezone_edit = 0;
static float longitude_edit = 0;
static float latitude_edit = 0;
static int32_t cursor_pos = 0;
static int32_t value_type = 0;
static wchar_t value_text[32] = L"00000000000";

// 将各类值转成可编辑的字符串
static void ui_app_setting_value_to_string(
    Key_Event *key_event, Global_State *global_state, wchar_t *value_text, int32_t value_type,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, float timezone, float longitude, float latitude
) {
    // 日期 yyyy-mm-dd
    if (value_type == 0) {
        value_text[0] = (wchar_t)get_digit(year, 3);
        value_text[1] = (wchar_t)get_digit(year, 2);
        value_text[2] = (wchar_t)get_digit(year, 1);
        value_text[3] = (wchar_t)get_digit(year, 0);
        value_text[4] = L'-';
        value_text[5] = (wchar_t)get_digit(month, 1);
        value_text[6] = (wchar_t)get_digit(month, 0);
        value_text[7] = L'-';
        value_text[8] = (wchar_t)get_digit(day, 1);
        value_text[9] = (wchar_t)get_digit(day, 0);
        value_text[10] = 0;
    }
    // 时间时区 hh:mmsaabb
    else if (value_type == 1) {
        value_text[0] = (wchar_t)get_digit(hour, 1);
        value_text[1] = (wchar_t)get_digit(hour, 0);
        value_text[2] = L':';
        value_text[3] = (wchar_t)get_digit(minute, 1);
        value_text[4] = (wchar_t)get_digit(minute, 0);
        value_text[5] = (wchar_t)get_timezone_digit(timezone, 0);
        value_text[6] = (wchar_t)get_timezone_digit(timezone, 1);
        value_text[7] = (wchar_t)get_timezone_digit(timezone, 2);
        value_text[8] = (wchar_t)get_timezone_digit(timezone, 3);
        value_text[9] = (wchar_t)get_timezone_digit(timezone, 4);
        value_text[10] = 0;
    }
    // 经度 sddd_mm'ss"
    else if (value_type == 2) {
        value_text[0] = (wchar_t)get_lon_lat_digit(longitude, 0);
        value_text[1] = (wchar_t)get_lon_lat_digit(longitude, 1);
        value_text[2] = (wchar_t)get_lon_lat_digit(longitude, 2);
        value_text[3] = (wchar_t)get_lon_lat_digit(longitude, 3);
        value_text[4] = L' ';
        value_text[5] = (wchar_t)get_lon_lat_digit(longitude, 4);
        value_text[6] = (wchar_t)get_lon_lat_digit(longitude, 5);
        value_text[7] = L'\'';
        value_text[8] = (wchar_t)get_lon_lat_digit(longitude, 6);
        value_text[9] = (wchar_t)get_lon_lat_digit(longitude, 7);
        value_text[10] = L'"';
        value_text[11] = 0;
    }
    // 纬度 sdd_mm'ss"
    else if (value_type == 3) {
        value_text[0] = (wchar_t)get_lon_lat_digit(latitude, 0);
        value_text[1] = (wchar_t)get_lon_lat_digit(latitude, 2);
        value_text[2] = (wchar_t)get_lon_lat_digit(latitude, 3);
        value_text[3] = L' ';
        value_text[4] = (wchar_t)get_lon_lat_digit(latitude, 4);
        value_text[5] = (wchar_t)get_lon_lat_digit(latitude, 5);
        value_text[6] = L'\'';
        value_text[7] = (wchar_t)get_lon_lat_digit(latitude, 6);
        value_text[8] = (wchar_t)get_lon_lat_digit(latitude, 7);
        value_text[9] = L'"';
        value_text[10] = 0;
    }
}

static void ui_app_setting_grid16_refresh_button(
    Key_Event *key_event, Global_State *global_state, int32_t is_single_line,
    int32_t col, int32_t row, wchar_t *text0, wchar_t *text1,
    uint8_t cell_bg_R, uint8_t cell_bg_G, uint8_t cell_bg_B, uint8_t cell_bg_mode,
    uint8_t cell_text0_R, uint8_t cell_text0_G, uint8_t cell_text0_B, uint8_t cell_text0_mode,
    uint8_t cell_text1_R, uint8_t cell_text1_G, uint8_t cell_text1_B, uint8_t cell_text1_mode
) {
    int32_t bx = (col == 0) ? 1 : 0;
    int32_t by = (row == 0) ? 1 : 0;
    // 网格布局：上下留白与顶/底栏高度一致（当前 UI 字体行高 + 1）
    const int32_t bar_height = gfx_font_line_height(global_state->ui_font) + 1;
    UI_Grid_Layout grid = ui_grid_layout_make(global_state->gfx->width, global_state->gfx->height, bar_height, bar_height, 0, 0, 4, 4);
    gfx_draw_rectangle(global_state->gfx, ui_grid_cell_x0(&grid,col)+bx, ui_grid_cell_y0(&grid,row)+by, ui_grid_cell_width(&grid)-1-bx, ui_grid_cell_height(&grid)-1-by, cell_bg_R, cell_bg_G, cell_bg_B, cell_bg_mode);
    if (is_single_line) {
        gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_16, text0, ui_grid_cell_center_x(&grid,col), ui_grid_cell_center_y(&grid,row), cell_text0_R, cell_text0_G, cell_text0_B, cell_text0_mode);
    }
    else {
        gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, text0, ui_grid_cell_center_x(&grid,col), ui_grid_cell_center_y(&grid,row)-8, cell_text0_R, cell_text0_G, cell_text0_B, cell_text0_mode);
        gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, text1, ui_grid_cell_center_x(&grid,col), ui_grid_cell_center_y(&grid,row)+10, cell_text1_R, cell_text1_G, cell_text1_B, cell_text1_mode);
    }
}

void ui_app_setting_grid16_draw(Key_Event *key_event, Global_State *global_state) {

    // 清屏
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_fill_white(global_state->gfx);
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_soft_clear(global_state->gfx);
    }

    uint8_t cell_bg_R = 0, cell_bg_G = 0, cell_bg_B = 0;
    uint8_t cell_text0_R = 0, cell_text0_G = 0, cell_text0_B = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        cell_bg_R = 233;
        cell_bg_G = 239;
        cell_bg_B = 255;
        cell_text0_R = 0;
        cell_text0_G = 0;
        cell_text0_B = 0;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        cell_bg_R = 40;
        cell_bg_G = 40;
        cell_bg_B = 42;
        cell_text0_R = 255;
        cell_text0_G = 255;
        cell_text0_B = 255;
    }

    wchar_t date_str[32];
    ui_app_setting_value_to_string(
        key_event, global_state, date_str, 0,
        global_state->year, global_state->month, global_state->day,
        global_state->hour, global_state->minute, global_state->timezone,
        global_state->longitude, global_state->latitude);

    wchar_t time_str[32];
    ui_app_setting_value_to_string(
        key_event, global_state, time_str, 1,
        global_state->year, global_state->month, global_state->day,
        global_state->hour, global_state->minute, global_state->timezone,
        global_state->longitude, global_state->latitude);

    wchar_t longitude_str[32];
    ui_app_setting_value_to_string(
        key_event, global_state, longitude_str, 2,
        global_state->year, global_state->month, global_state->day,
        global_state->hour, global_state->minute, global_state->timezone,
        global_state->longitude, global_state->latitude);

    wchar_t latitude_str[32];
    ui_app_setting_value_to_string(
        key_event, global_state, latitude_str, 3,
        global_state->year, global_state->month, global_state->day,
        global_state->hour, global_state->minute, global_state->timezone,
        global_state->longitude, global_state->latitude);

    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        0, 0, L"日期", date_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        1, 0, L"时间", time_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);

    wchar_t brightness_str[32];
    swprintf(brightness_str, 32, L"%d%%", (global_state->brightness * 100) / 255);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        2, 0, L"屏幕亮度", brightness_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        3, 0, L"返回", NULL, cell_bg_R+10, cell_bg_G+10, cell_bg_B+10, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);

    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        0, 1, L"经度", longitude_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        1, 1, L"纬度", latitude_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);

    wchar_t volume_str[32];
    swprintf(volume_str, 32, L"%d%%", (global_state->volume * 100) / 255);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        2, 1, L"音量", volume_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        3, 1, L"IMU", L"开", cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0x00, 1);

    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        0, 2, L"LLM演示",
        (global_state->llm_enable_observation == 0) ? L"关闭" : L"开启",
        cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        1, 2, L"TTS设置",
        (global_state->tts_req_mode == 0) ? L"关闭" : ((global_state->tts_req_mode == 1) ? L"实时转换" : L"统一转换"),
        cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        2, 2, L"ASR设置",
        (global_state->is_auto_submit_after_asr == 0) ? L"编辑后提交" : L"立刻提交",
        cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    wchar_t auto_shutdown_str[32];
    if (global_state->auto_shutdown_minutes <= 0) {
        wcscpy(auto_shutdown_str, L"关");
    }
    else {
        swprintf(auto_shutdown_str, 32, L"%d分钟", (int)global_state->auto_shutdown_minutes);
    }

    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        3, 2, L"自动关机", auto_shutdown_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0x00, 0x00, 1);

    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        1, 3, L"颜色风格", (global_state->ui_color_style == UI_COLOR_LIGHT) ? L"亮" : L"暗",
        cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    // 时钟样式（启动画面：小=时间并入顶部日期行；大=七段码钟居画面中央）
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        0, 3, L"时钟样式", (global_state->splash_clock_style == SPLASH_CLOCK_SMALL) ? L"小" : L"大",
        cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    // 按键提示（按键反馈方式：无/灯光/蜂鸣/灯光+蜂鸣）
    {
        static const wchar_t *KEY_FEEDBACK_MODE_STR[] = {L"无", L"灯光", L"蜂鸣", L"灯光+蜂鸣"};
        int32_t kf_mode = global_state->key_feedback_mode & 3;
        ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
            2, 3, L"按键提示", KEY_FEEDBACK_MODE_STR[kf_mode], cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    }

    // 顶栏：十六宫格设置窗口专用，固定 1 倍字体行高 + 1（与网格布局留白一致；
    // 非文本显示控件页眉，不随标准页眉 1.5 倍行高变动）
    ui_draw_header_ex(key_event, global_state, L"系统设置", 1, gfx_font_line_height(global_state->ui_font) + 1);
    ui_draw_footer(key_event, global_state, L"(c) 2025-2026 BD4SUR", 1);
}

static inline void ui_app_setting_capture_value(Key_Event *key_event, Global_State *global_state) {
    year_edit = global_state->year;
    month_edit = global_state->month;
    day_edit = global_state->day;
    hour_edit = global_state->hour;
    minute_edit = global_state->minute;
    timezone_edit = global_state->timezone;
    longitude_edit = global_state->longitude;
    latitude_edit = global_state->latitude;
}

void ui_app_setting_grid16_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 日期
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_1) {
        value_type = 0;
        ui_app_setting_capture_value(key_event, global_state);
        ui_app_setting_value_to_string(
            key_event, global_state, value_text, value_type,
            year_edit, month_edit, day_edit, hour_edit, minute_edit, timezone_edit, longitude_edit, latitude_edit);
        global_state->STATE = STATE_SETTING_INPUT;
    }
    // 时间和时区
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_2) {
        value_type = 1;
        ui_app_setting_capture_value(key_event, global_state);
        ui_app_setting_value_to_string(
            key_event, global_state, value_text, value_type,
            year_edit, month_edit, day_edit, hour_edit, minute_edit, timezone_edit, longitude_edit, latitude_edit);
        global_state->STATE = STATE_SETTING_INPUT;
    }
    // 屏幕亮度
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_3) {
        global_state->brightness += 32;
        if (global_state->brightness == 256) {
            global_state->brightness = 255; // 最高亮度挡位
        }
        else if (global_state->brightness > 256) {
            global_state->brightness = 32; // 不允许亮度为0，回绕到最低挡位
        }
        gfx_set_brightness(global_state->gfx, global_state->brightness);
    }
    // 经度
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_4) {
        value_type = 2;
        ui_app_setting_capture_value(key_event, global_state);
        ui_app_setting_value_to_string(
            key_event, global_state, value_text, value_type,
            year_edit, month_edit, day_edit, hour_edit, minute_edit, timezone_edit, longitude_edit, latitude_edit);
        global_state->STATE = STATE_SETTING_INPUT;
    }
    // 纬度
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_5) {
        value_type = 3;
        ui_app_setting_capture_value(key_event, global_state);
        ui_app_setting_value_to_string(
            key_event, global_state, value_text, value_type,
            year_edit, month_edit, day_edit, hour_edit, minute_edit, timezone_edit, longitude_edit, latitude_edit);
        global_state->STATE = STATE_SETTING_INPUT;
    }
    // 音量（全局主音量：+16 步进；=256 钳位 255；>256 回绕 0 静音；影响按键音/寻呼机发射/音乐盒初始音量）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_6) {
        global_state->volume += 16;
        if (global_state->volume == 256) {
            global_state->volume = 255; // 最高音量挡位
        }
        else if (global_state->volume > 256) {
            global_state->volume = 0; // 回绕到静音
        }
        audio_out_set_master_volume((uint8_t)global_state->volume);
    }
    // LLM演示设置
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_7) {
        global_state->llm_enable_observation += 1;
        global_state->llm_enable_observation = global_state->llm_enable_observation % 2;
    }
    // TTS设置
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_8) {
        global_state->tts_req_mode += 1;
        global_state->tts_req_mode = global_state->tts_req_mode % 3;
    }
    // ASR设置
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_9) {
        global_state->is_auto_submit_after_asr += 1;
        global_state->is_auto_submit_after_asr = global_state->is_auto_submit_after_asr % 2;
    }
    // 颜色风格（全局UI色彩风格：亮/暗切换，默认暗；重绘由下方统一刷新块完成，
    // 不得在此绘制主菜单宫格——此前误调用 ui_widget_grid16_draw 导致点击后闪一下主菜单）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_0) {
        if (global_state->ui_color_style == UI_COLOR_LIGHT) {
            global_state->ui_color_style = UI_COLOR_DARK;
        }
        else {
            global_state->ui_color_style = UI_COLOR_LIGHT;
        }
    }
    // 时钟样式（启动画面时钟：小↔大切换；十六宫格左下角键 = NANO_KEY_left）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_left) {
        global_state->splash_clock_style = (global_state->splash_clock_style == SPLASH_CLOCK_SMALL)
            ? SPLASH_CLOCK_BIG : SPLASH_CLOCK_SMALL;
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_SPLASH_SCREEN;
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_shift) {
        // TODO
    }
    // 自动关机（关→1→2→3→5→10→20→30→60 分钟循环；设置即从当前时刻重新倒计时）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_ctrl) {
        static const int32_t AUTO_SHUTDOWN_OPTIONS[] = {0, 1, 2, 3, 5, 10, 20, 30, 60};
        int32_t idx = 0;
        for (int32_t i = 0; i < (int32_t)(sizeof(AUTO_SHUTDOWN_OPTIONS) / sizeof(AUTO_SHUTDOWN_OPTIONS[0])); i++) {
            if (AUTO_SHUTDOWN_OPTIONS[i] == global_state->auto_shutdown_minutes) { idx = i; break; }
        }
        idx = (idx + 1) % (int32_t)(sizeof(AUTO_SHUTDOWN_OPTIONS) / sizeof(AUTO_SHUTDOWN_OPTIONS[0]));
        global_state->auto_shutdown_minutes = AUTO_SHUTDOWN_OPTIONS[idx];
        global_state->auto_shutdown_deadline = (AUTO_SHUTDOWN_OPTIONS[idx] > 0)
            ? global_state->timestamp + (uint64_t)AUTO_SHUTDOWN_OPTIONS[idx] * 60000ULL : 0;
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_enter) {
        // TODO
    }
    // 按键提示（按键反馈方式：无→灯光→蜂鸣→灯光+蜂鸣 循环）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_right) {
        global_state->key_feedback_mode = (global_state->key_feedback_mode + 1) % 4;
    }
    else {
        return;
    }

    // 有键按下则刷新
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code != NANO_KEY_IDLE) {
        ui_app_setting_grid16_draw(key_event, global_state);
        gfx_refresh(global_state->gfx);
    }
}

// 日期/时间/经度/纬度设置
// value_type: 0-日期 1-时间时区 2-经度 3-纬度
// cursor_pos: 光标相对于值字符串第一个字符的位置（不检测连字符等非值字符，位置由调用者处理），例如:
//   value_str   12:34+0800
//   cursor_pos  0123456789
void ui_app_setting_value_input_draw(Key_Event *key_event, Global_State *global_state, int32_t value_type, wchar_t *value_text, int32_t cursor_pos) {
    // 清屏
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_fill_white(global_state->gfx);
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_soft_clear(global_state->gfx);
    }

    uint8_t cell_bg_R = 0, cell_bg_G = 0, cell_bg_B = 0;
    uint8_t cell_text0_R = 0, cell_text0_G = 0, cell_text0_B = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        cell_bg_R = 233;
        cell_bg_G = 239;
        cell_bg_B = 255;
        cell_text0_R = 0;
        cell_text0_G = 0;
        cell_text0_B = 0;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        cell_bg_R = 40;
        cell_bg_G = 40;
        cell_bg_B = 42;
        cell_text0_R = 255;
        cell_text0_G = 255;
        cell_text0_B = 255;
    }

    // 绘制顶栏（前缀；页眉带内垂直居中）
    int32_t header_h = ui_std_header_height(global_state->ui_font);
    int32_t header_text_y = (header_h - gfx_font_line_height(GFX_FONT_BITMAP_12)) / 2;
    if (header_text_y < 0) header_text_y = 0;
    switch (value_type) {
        case 0: {
            ui_draw_header(key_event, global_state, L"", 0);
            gfx_font_draw_text(global_state->gfx, GFX_FONT_BITMAP_12, L"设置时间：", 0, header_text_y, 255, 255, 255, 1);
            break;
        }
        case 1: {
            ui_draw_header(key_event, global_state, L"", 0);
            gfx_font_draw_text(global_state->gfx, GFX_FONT_BITMAP_12, L"设置日期：", 0, header_text_y, 255, 255, 255, 1);
            break;
        }
        case 2: {
            ui_draw_header(key_event, global_state, L"", 0);
            gfx_font_draw_text(global_state->gfx, GFX_FONT_BITMAP_12, L"设置经度：", 0, header_text_y, 255, 255, 255, 1);
            break;
        }
        case 3: {
            ui_draw_header(key_event, global_state, L"", 0);
            gfx_font_draw_text(global_state->gfx, GFX_FONT_BITMAP_12, L"设置纬度：", 0, header_text_y, 255, 255, 255, 1);
            break;
        }
        default: return;
    }

    // 绘制设置值和光标
    int32_t x0 = 12 * 5; // 与顶栏前缀的长度有关
    int32_t x_cur = x0 + cursor_pos * 6;
    gfx_font_draw_text(global_state->gfx, GFX_FONT_BITMAP_12, value_text, x0, header_text_y, 0x00, 0xff, 0xff, 1);
    gfx_draw_rectangle(global_state->gfx, x_cur, header_text_y + 11, 5, 2, 0x00, 0xff, 0xff, 1);

    // 绘制底栏
    ui_draw_footer(key_event, global_state, L"按数字键输入 光标自动右移", 1);

    // 绘制按键
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        0, 0, L"1", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        1, 0, L"2", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        2, 0, L"3", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        3, 0, L"取消", NULL, cell_bg_R+10, cell_bg_G, cell_bg_B, 1, 0xff, 0x00, 0x00, 1, 0x00, 0x00, 0x00, 1);

    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        0, 1, L"4", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        1, 1, L"5", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        2, 1, L"6", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    switch (value_type) {
        case 0: break;
        case 1:
            if (cursor_pos == 5) {
                ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
                    3, 1, L"东(+)", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
            }
            break;
        case 2:
            if (cursor_pos == 0) {
                ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
                    3, 1, L"东经(+)", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
            }
            break;
        case 3:
            if (cursor_pos == 0) {
                ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
                    3, 1, L"北纬(+)", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
            }
            break;
        default: return;
    }


    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        0, 2, L"7", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        1, 2, L"8", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        2, 2, L"9", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    switch (value_type) {
        case 0: break;
        case 1:
            if (cursor_pos == 5) {
                ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
                    3, 2, L"西(-)", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
            }
            break;
        case 2:
            if (cursor_pos == 0) {
                ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
                    3, 2, L"西经(-)", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
            }
            break;
        case 3:
            if (cursor_pos == 0) {
                ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
                    3, 2, L"南纬(-)", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
            }
            break;
        default: return;
    }

    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        0, 3, L"←", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        1, 3, L"0", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        2, 3, L"→", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        3, 3, L"确认", NULL, cell_bg_R, cell_bg_G+10, cell_bg_B, 1, 0x00, 0xff, 0x00, 1, 0x00, 0x00, 0x00, 1);
}


// 计算下个光标位置
static int32_t ui_app_setting_next_pos(Key_Event *key_event, Global_State *global_state, int32_t value_type, int32_t current_pos) {
    // yyyy-mm-dd
    // 0123456789    
    if (value_type == 0) {
        switch (current_pos) {
            case 0: return 1; break;
            case 1: return 2; break;
            case 2: return 3; break;
            case 3: return 5; break;
            case 4: return 5; break;
            case 5: return 6; break;
            case 6: return 8; break;
            case 7: return 8; break;
            case 8: return 9; break;
            case 9: return 0; break;
            default: return 0; break;
        }
    }
    // hh:mmsaabb
    // 0123456789
    else if (value_type == 1) {
        switch (current_pos) {
            case 0: return 1; break;
            case 1: return 3; break;
            case 2: return 3; break;
            case 3: return 4; break;
            case 4: return 5; break;
            case 5: return 6; break;
            case 6: return 7; break;
            case 7: return 8; break;
            case 8: return 9; break;
            case 9: return 0; break;
            default: return 0; break;
        }
    }
    // 经度
    // sddd_mm'ss"
    // 0123456789A
    else if (value_type == 2) {
        switch (current_pos) {
            case 0: return 1; break;
            case 1: return 2; break;
            case 2: return 3; break;
            case 3: return 5; break;
            case 4: return 5; break;
            case 5: return 6; break;
            case 6: return 8; break;
            case 7: return 8; break;
            case 8: return 9; break;
            case 9: return 0; break;
            default: return 0; break;
        }
    }
    // 纬度
    // sdd_mm'ss"
    // 0123456789
    else if (value_type == 3) {
        switch (current_pos) {
            case 0: return 1; break;
            case 1: return 2; break;
            case 2: return 4; break;
            case 3: return 4; break;
            case 4: return 5; break;
            case 5: return 7; break;
            case 6: return 7; break;
            case 7: return 8; break;
            case 8: return 0; break;
            default: return 0; break;
        }
    }
    else return 0;
}

// 计算上个光标位置
static int32_t ui_app_setting_prev_pos(Key_Event *key_event, Global_State *global_state, int32_t value_type, int32_t current_pos) {
    // yyyy-mm-dd
    // 0123456789    
    if (value_type == 0) {
        switch (current_pos) {
            case 0: return 9; break;
            case 1: return 0; break;
            case 2: return 1; break;
            case 3: return 2; break;
            case 4: return 3; break;
            case 5: return 3; break;
            case 6: return 5; break;
            case 7: return 6; break;
            case 8: return 6; break;
            case 9: return 8; break;
            default: return 9; break;
        }
    }
    // hh:mmsaabb
    // 0123456789
    else if (value_type == 1) {
        switch (current_pos) {
            case 0: return 9; break;
            case 1: return 0; break;
            case 2: return 1; break;
            case 3: return 1; break;
            case 4: return 3; break;
            case 5: return 4; break;
            case 6: return 5; break;
            case 7: return 6; break;
            case 8: return 7; break;
            case 9: return 8; break;
            default: return 9; break;
        }
    }
    // 经度
    // sddd_mm'ss"
    // 0123456789A
    else if (value_type == 2) {
        switch (current_pos) {
            case 0: return 9; break;
            case 1: return 0; break;
            case 2: return 1; break;
            case 3: return 2; break;
            case 4: return 3; break;
            case 5: return 3; break;
            case 6: return 5; break;
            case 7: return 6; break;
            case 8: return 6; break;
            case 9: return 8; break;
            default: return 9; break;
        }
    }
    // 纬度
    // sdd_mm'ss"
    // 0123456789
    else if (value_type == 3) {
        switch (current_pos) {
            case 0: return 8; break;
            case 1: return 0; break;
            case 2: return 1; break;
            case 3: return 2; break;
            case 4: return 2; break;
            case 5: return 4; break;
            case 6: return 5; break;
            case 7: return 5; break;
            case 8: return 7; break;
            default: return 8; break;
        }
    }
    else return 0;
}

void ui_app_setting_value_input_event_handler(Key_Event *key_event, Global_State *global_state, int32_t value_type) {
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_1) {
        value_text[cursor_pos] = L'1';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_2) {
        value_text[cursor_pos] = L'2';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_3) {
        value_text[cursor_pos] = L'3';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_4) {
        value_text[cursor_pos] = L'4';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_5) {
        value_text[cursor_pos] = L'5';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_6) {
        value_text[cursor_pos] = L'6';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_7) {
        value_text[cursor_pos] = L'7';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_8) {
        value_text[cursor_pos] = L'8';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_9) {
        value_text[cursor_pos] = L'9';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_0) {
        value_text[cursor_pos] = L'0';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_SETTING_MENU;
        return;
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_shift) {
        value_text[cursor_pos] = L'+';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_ctrl) {
        value_text[cursor_pos] = L'-';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_enter) {
        // 日期 yyyy-mm-dd
        if (value_type == 0) {
            global_state->year =(value_text[0] - L'0') * 1000 + 
                                (value_text[1] - L'0') * 100 + 
                                (value_text[2] - L'0') * 10 + 
                                (value_text[3] - L'0') * 1;
            global_state->month=(value_text[5] - L'0') * 10 + 
                                (value_text[6] - L'0') * 1;
            global_state->day = (value_text[8] - L'0') * 10 + 
                                (value_text[9] - L'0') * 1;

            int32_t utc_year = 0, utc_month = 0, utc_day = 0, utc_hour = 0, utc_minute = 0, utc_second = 0;
            local_time_to_utc(
                global_state->year, global_state->month, global_state->day,
                global_state->hour, global_state->minute, global_state->second,
                global_state->timezone,
                &utc_year, &utc_month, &utc_day, &utc_hour, &utc_minute, &utc_second
            );
            set_sys_time(utc_year, utc_month, utc_day, utc_hour, utc_minute, 0);
        }
        // 时间时区 hh:mmsaabb
        else if (value_type == 1) {
            global_state->hour =    (value_text[0] - L'0') * 10 + 
                                    (value_text[1] - L'0') * 1;
            global_state->minute =  (value_text[3] - L'0') * 10 + 
                                    (value_text[4] - L'0') * 1;
            float tz_sign =         (value_text[5] == '+') ? 1.0f : (-1.0f);
            float tz_hour =         (value_text[6] - '0') * 10.0f + 
                                    (value_text[7] - '0') * 1.0f;
            float tz_min =          (value_text[8] - '0') * 10.0f + 
                                    (value_text[9] - '0') * 1.0f;
            global_state->timezone = tz_sign * (tz_hour + tz_min / 60.0f);

            int32_t utc_year = 0, utc_month = 0, utc_day = 0, utc_hour = 0, utc_minute = 0, utc_second = 0;
            local_time_to_utc(
                global_state->year, global_state->month, global_state->day,
                global_state->hour, global_state->minute, global_state->second,
                global_state->timezone,
                &utc_year, &utc_month, &utc_day, &utc_hour, &utc_minute, &utc_second
            );
            set_sys_time(utc_year, utc_month, utc_day, utc_hour, utc_minute, 0);
        }
        // 经度 sddd_mm'ss"
        else if (value_type == 2) {
            float lon_sign= (value_text[0] == '+') ? 1.0f : (-1.0f);
            float lon_hour= (value_text[1] - '0') * 100 + 
                            (value_text[2] - '0') * 10 + 
                            (value_text[3] - '0') * 1;
            float lon_min = (value_text[5] - '0') * 10 + 
                            (value_text[6] - '0') * 1;
            float lon_sec = (value_text[8] - '0') * 10 + 
                            (value_text[9] - '0') * 1;
            global_state->longitude = lon_sign * (lon_hour + lon_min / 60.0f + lon_sec / 3600.0f);
        }
        // 纬度 sdd_mm'ss"
        else if (value_type == 3) {
            float lat_sign= (value_text[0] == '+') ? 1.0f : (-1.0f);
            float lat_hour= (value_text[1] - '0') * 10 + 
                            (value_text[2] - '0') * 1;
            float lat_min = (value_text[4] - '0') * 10 + 
                            (value_text[5] - '0') * 1;
            float lat_sec = (value_text[7] - '0') * 10 + 
                            (value_text[8] - '0') * 1;
            global_state->latitude = lat_sign * (lat_hour + lat_min / 60.0f + lat_sec / 3600.0f);
        }
        global_state->STATE = STATE_SETTING_MENU;
        return;
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_left) {
        cursor_pos = ui_app_setting_prev_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_right) {
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }

    // 有键按下则刷新
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code != NANO_KEY_IDLE) {
        ui_app_setting_value_input_draw(key_event, global_state, value_type, value_text, cursor_pos);
        gfx_refresh(global_state->gfx);
    }
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
    // 初始化玲珑天象仪

    global_state->linglong_cfg = (Linglong_Config *)platform_calloc(1, sizeof(Linglong_Config));
    linglong_init(global_state->linglong_cfg);

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    global_state->llgfx = (Nano_GFX*)platform_calloc(1, sizeof(Nano_GFX));
    global_state->llgfx->is_double_buffer = 0;
    gfx_init(global_state->llgfx, SCREEN_WIDTH, SCREEN_HEIGHT, GFX_COLOR_MODE_RGB888);
#else
    global_state->llgfx = global_state->gfx;
#endif


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
        case STATE_SPECTROGRAM:      // 关麦+释放工作区 → 主菜单
            ui_spectrogram_deinit(key_event, global_state);
            global_state->STATE = STATE_MAIN_MENU;
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

// 退出确认模态框几何（居中圆角对话框 + 两个圆角按钮）——通用组件，
// 供控制台退出、俄罗斯方块退出、以及灵机 App 的 (Ui.confirm)/(App.exit) 共用
#define UI_ANIMAC_EXIT_DIALOG_W   (200)
#define UI_ANIMAC_EXIT_DIALOG_H   (96)
#define UI_ANIMAC_EXIT_DIALOG_H_MSG (110)  // 带正文时加高一行（12px 字模）
#define UI_ANIMAC_EXIT_BTN_W      (84)   // 两按钮（控制台/俄罗斯方块等 C 场景）
#define UI_ANIMAC_EXIT_BTN_W3     (54)   // 三按钮（灵机 App：确认/后台/留下）
#define UI_ANIMAC_EXIT_BTN_H      (30)
#define UI_ANIMAC_EXIT_BTN_GAP    (12)
#define UI_ANIMAC_EXIT_BTN_GAP3   (8)

// 几何计算（内部共用；with_body 决定对话框高度，with_bg 决定按钮行是两枚还是三枚）
static void ui_exit_confirm_geom(Global_State *global_state, int32_t with_body, int32_t with_bg,
    int32_t *out_dx, int32_t *out_dy, int32_t *out_h, int32_t *out_btn_y,
    int32_t *out_btn_w, int32_t *out_btn_gap,
    int32_t *out_confirm_x, int32_t *out_bg_x, int32_t *out_stay_x
) {
    int32_t h  = with_body ? UI_ANIMAC_EXIT_DIALOG_H_MSG : UI_ANIMAC_EXIT_DIALOG_H;
    int32_t dx = ((int32_t)global_state->gfx->width - UI_ANIMAC_EXIT_DIALOG_W) / 2;
    int32_t dy = ((int32_t)global_state->gfx->height - h) / 2;
    int32_t bw  = with_bg ? UI_ANIMAC_EXIT_BTN_W3 : UI_ANIMAC_EXIT_BTN_W;
    int32_t gap = with_bg ? UI_ANIMAC_EXIT_BTN_GAP3 : UI_ANIMAC_EXIT_BTN_GAP;
    int32_t n   = with_bg ? 3 : 2;
    int32_t confirm_x = dx + (UI_ANIMAC_EXIT_DIALOG_W - (bw * n + gap * (n - 1))) / 2;

    *out_dx = dx;
    *out_dy = dy;
    *out_h = h;
    *out_btn_y = dy + (with_body ? 66 : 54);
    *out_btn_w = bw;
    *out_btn_gap = gap;
    *out_confirm_x = confirm_x;
    *out_bg_x = confirm_x + (bw + gap);          // 仅 with_bg 有效
    *out_stay_x = confirm_x + (bw + gap) * (n - 1);
}

// 模态框按钮排布：返回对话框左上角与按钮区基准（供绘制与命中判定共用，保证严格一致）
void ui_exit_confirm_layout(Global_State *global_state,
    int32_t *out_dx, int32_t *out_dy, int32_t *out_btn_y, int32_t *out_confirm_x, int32_t *out_stay_x
) {
    int32_t h = 0, bw = 0, gap = 0, bgx = 0;
    ui_exit_confirm_geom(global_state, 0, 0, out_dx, out_dy, &h, out_btn_y, &bw, &gap, out_confirm_x, &bgx, out_stay_x);
}

// 按可用宽度截断一行文本（超出时以省略号结尾；从尾部逐字删，不改动前述码点）
static void ui_exit_confirm_truncate(uint32_t font_id, wchar_t *line, size_t cap, int32_t max_w) {
    size_t len = wcslen(line);
    if (len == 0 || cap < 3) return;
    if (gfx_font_measure_text(font_id, line) <= max_w) return;
    if (len > cap - 2) len = cap - 2;   // 保证 line[len+1] 不越界
    while (len > 0) {
        line[len] = L'\u2026';
        line[len + 1] = L'\0';
        if (gfx_font_measure_text(font_id, line) <= max_w) return;
        len--;
    }
    line[0] = L'\0';
}

// 绘制退出确认模态框（叠加在当前画面之上；全部元素圆角矩形）
void ui_exit_confirm_draw(Key_Event *key_event, Global_State *global_state) {
    ui_exit_confirm_draw_msg(key_event, global_state, L"是否退出？", NULL);
}

// 绘制可定制标题/正文的确认模态框（(Ui.confirm ...) 与 (native App) 共用）
void ui_exit_confirm_draw_msg(Key_Event *key_event, Global_State *global_state,
                              const wchar_t *title, const wchar_t *body) {
    ui_exit_confirm_draw_msg_ex(key_event, global_state, title, body, 0);
}

// 三按钮变体：with_bg=1 时按钮行是「确认」「后台」「留下」
void ui_exit_confirm_draw_msg_ex(Key_Event *key_event, Global_State *global_state,
                                 const wchar_t *title, const wchar_t *body, int32_t with_bg) {
    (void)key_event;
    Nano_GFX *gfx = global_state->gfx;
    int32_t dx = 0, dy = 0, h = 0, btn_y = 0, confirm_x = 0, bg_x = 0, stay_x = 0;
    int32_t btn_w = 0, btn_gap = 0;
    int32_t has_body = (body != NULL && body[0] != L'\0');

    ui_exit_confirm_geom(global_state, has_body, with_bg, &dx, &dy, &h, &btn_y, &btn_w, &btn_gap,
                         &confirm_x, &bg_x, &stay_x);

    // 配色随全局色彩风格
    uint8_t border_R = 70,  border_G = 70,  border_B = 78;    // 对话框描边
    uint8_t dlg_R = 24,     dlg_G = 24,     dlg_B = 28;       // 对话框底色
    uint8_t btn_R = 46,     btn_G = 46,     btn_B = 50;       // “留下”按钮底色
    uint8_t ok_R = 16,      ok_G = 72,      ok_B = 176;       // “确认”按钮底色（强调色，同 Ctrl 高亮色系）
    uint8_t text_R = 240,   text_G = 240,   text_B = 240;     // 文字
    uint8_t stay_text_R = 240, stay_text_G = 240, stay_text_B = 240;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        border_R = 180; border_G = 180; border_B = 190;
        dlg_R = 250; dlg_G = 250; dlg_B = 252;
        btn_R = 224; btn_G = 230; btn_B = 234;
        ok_R = 17; ok_G = 85; ok_B = 238;
        text_R = 17; text_G = 17; text_B = 17;
        stay_text_R = 17; stay_text_G = 17; stay_text_B = 17;
    }

    // 描边 + 底色两层圆角矩形（外圈 2px 描边效果）
    gfx_draw_rounded_rectangle(gfx, dx - 2, dy - 2, UI_ANIMAC_EXIT_DIALOG_W + 4, h + 4,
        8, 8, 8, 8, border_R, border_G, border_B, 1);
    gfx_draw_rounded_rectangle(gfx, dx, dy, UI_ANIMAC_EXIT_DIALOG_W, h,
        6, 6, 6, 6, dlg_R, dlg_G, dlg_B, 1);

    // 标题（调用方未给标题时退回历史文案）
    {
        wchar_t title_buf[40];
        const wchar_t *t = (title != NULL && title[0] != L'\0') ? title : L"是否退出？";
        wcsncpy(title_buf, t, sizeof(title_buf) / sizeof(title_buf[0]) - 1);
        title_buf[sizeof(title_buf) / sizeof(title_buf[0]) - 1] = L'\0';
        ui_exit_confirm_truncate(GFX_FONT_ALPHA_16, title_buf, sizeof(title_buf) / sizeof(title_buf[0]),
                                 UI_ANIMAC_EXIT_DIALOG_W - 16);
        gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_16, title_buf,
            (int32_t)gfx->width / 2, dy + 26, text_R, text_G, text_B, 1);
    }

    // 正文（可选；12px 字模，单行居中，超宽截断）
    if (has_body) {
        wchar_t body_buf[64];
        wcsncpy(body_buf, body, sizeof(body_buf) / sizeof(body_buf[0]) - 1);
        body_buf[sizeof(body_buf) / sizeof(body_buf[0]) - 1] = L'\0';
        ui_exit_confirm_truncate(GFX_FONT_ALPHA_12, body_buf, sizeof(body_buf) / sizeof(body_buf[0]),
                                 UI_ANIMAC_EXIT_DIALOG_W - 12);
        gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, body_buf,
            (int32_t)gfx->width / 2, dy + 46, text_R, text_G, text_B, 1);
    }

    // 按钮：确认（强调色） / [后台] / 留下（中性色）
    // 三按钮行比两按钮行窄，故字号降一档（12px）以免"后台"挤不下
    {
        uint32_t fnt = with_bg ? GFX_FONT_ALPHA_12 : GFX_FONT_ALPHA_16;
        int32_t ty = btn_y + UI_ANIMAC_EXIT_BTN_H / 2 - (with_bg ? 1 : 0);
        gfx_draw_rounded_rectangle(gfx, confirm_x, btn_y, btn_w, UI_ANIMAC_EXIT_BTN_H,
            6, 6, 6, 6, ok_R, ok_G, ok_B, 1);
        gfx_font_draw_text_centered(gfx, fnt, L"确认",
            confirm_x + btn_w / 2, ty, 240, 240, 240, 1);
        if (with_bg) {
            gfx_draw_rounded_rectangle(gfx, bg_x, btn_y, btn_w, UI_ANIMAC_EXIT_BTN_H,
                6, 6, 6, 6, btn_R, btn_G, btn_B, 1);
            gfx_font_draw_text_centered(gfx, fnt, L"后台",
                bg_x + btn_w / 2, ty, stay_text_R, stay_text_G, stay_text_B, 1);
        }
        gfx_draw_rounded_rectangle(gfx, stay_x, btn_y, btn_w, UI_ANIMAC_EXIT_BTN_H,
            6, 6, 6, 6, btn_R, btn_G, btn_B, 1);
        gfx_font_draw_text_centered(gfx, fnt, L"留下",
            stay_x + btn_w / 2, ty, stay_text_R, stay_text_G, stay_text_B, 1);
    }

    gfx_refresh(gfx);
}

// 模态框命中判定：1=确认 2=留下 0=未命中
int32_t ui_exit_confirm_hit(Global_State *global_state, int32_t x, int32_t y) {
    return ui_exit_confirm_hit_msg(global_state, 0, x, y);
}

// 带正文版命中判定（with_body 须与绘制时一致）
int32_t ui_exit_confirm_hit_msg(Global_State *global_state, int32_t with_body, int32_t x, int32_t y) {
    return ui_exit_confirm_hit_msg_ex(global_state, with_body, 0, x, y);
}

// 三按钮变体命中判定：1=确认 2=留下 3=后台 0=未命中（with_body/with_bg 须与绘制时一致）
int32_t ui_exit_confirm_hit_msg_ex(Global_State *global_state, int32_t with_body, int32_t with_bg,
                                   int32_t x, int32_t y) {
    int32_t dx = 0, dy = 0, h = 0, btn_y = 0, confirm_x = 0, bg_x = 0, stay_x = 0;
    int32_t btn_w = 0, btn_gap = 0;
    ui_exit_confirm_geom(global_state, with_body, with_bg, &dx, &dy, &h, &btn_y, &btn_w, &btn_gap,
                         &confirm_x, &bg_x, &stay_x);
    (void)dx; (void)dy; (void)h; (void)btn_gap;
    if (y >= btn_y && y < btn_y + UI_ANIMAC_EXIT_BTN_H) {
        if (x >= confirm_x && x < confirm_x + btn_w) return 1;
        if (with_bg && x >= bg_x && x < bg_x + btn_w) return 3;
        if (x >= stay_x && x < stay_x + btn_w) return 2;
    }
    return 0;
}


// ===============================================================================
// 安全关机：手机式「滑动关机」滑块
//
// 交互：抓住轨道左端的圆形推钮向右拖动，滑到右端（进度 ≥ UI_SHUTDOWN_COMPLETE_PCT）
// 后松手才真正调用 graceful_shutdown()；未滑到位松手则回弹到起点。触屏只认推钮
// （含少量触摸余量），点按轨道其余位置不会误触发。本状态已加入
// ui_app_state_is_menu 抑制表，十六宫格软键整体退场；实体键盘保留原有快捷方式：
// 长按 D 关机、A 取消。
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

    // 5. 推钮内的右向箭头：抗锯齿线段（叠加混合，白线在红色胶囊上自然过渡）
    {
        int32_t s = r / 2;                   // 箭头半高
        if (s < 4) s = 4;
        const float ax = (float)knob_cx + (float)s * 0.15f; // 视觉居中略偏右一点点
        const float ay = (float)cy;
        const float lw = (r >= 18) ? 3.0f : 2.0f;
        gfx_draw_line_anti_aliasing(gfx, ax - (float)s * 0.5f, ay - (float)s,
            ax + (float)s * 0.5f, ay, lw,
            p.on_accent[0], p.on_accent[1], p.on_accent[2], 1);
        gfx_draw_line_anti_aliasing(gfx, ax + (float)s * 0.5f, ay,
            ax - (float)s * 0.5f, ay + (float)s, lw,
            p.on_accent[0], p.on_accent[1], p.on_accent[2], 1);
    }
}

// 页眉高度（整页与提示页共用；页眉右侧「返回」热点也以此判定）
static int32_t ui_shutdown_header_h(Global_State *global_state) {
    return ui_std_header_height(global_state->ui_font);
}

// 整页重绘（清屏 + 页眉 / 页脚 + 滑块）。页眉右侧固定「返回」热点：触屏设备
// （无硬键盘）在关机页的取消入口，避免进入后无法退出（十六宫格软键已整体抑制）。
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

        // 按下任何键，不论长短按，进入主菜单
        if (key_event->key_edge < 0 && key_event->key_code != NANO_KEY_IDLE) {
            global_state->STATE = STATE_MAIN_MENU;
        }

        break;

    /////////////////////////////////////////////
    // 主菜单。
    /////////////////////////////////////////////

    case STATE_MAIN_MENU:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_widget_grid16_draw(key_event, global_state);
            gfx_refresh(global_state->gfx);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_widget_grid16_event_handler(key_event, global_state);

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
    // 日历
    /////////////////////////////////////////////

    case STATE_CALENDAR:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_calendar_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_calendar_render_frame(key_event, global_state);
        ui_calendar_event_handler(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 设置菜单
    /////////////////////////////////////////////

    case STATE_SETTING_MENU:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_app_setting_grid16_draw(key_event, global_state);
            gfx_refresh(global_state->gfx);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_app_setting_grid16_event_handler(key_event, global_state);

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
            swprintf(readme, 1024, L"[#66ccff]Project Linglong%ls v" NANO_VERSION "\n玲珑电子核桃 | M5Core2(ESP32)\n(c) 2025-2026 BD4SUR\ngithub.com/bd4sur/Nano\n\n# 概述\n\n本机是一款“电子核桃”，试图在各种便携设备上实现尽量多、尽量有趣、尽量实用的功能。固件以C语言编写，可在不同的计算设备上部署运行。图形框架为自主实现，并嵌入了自制的灵机(Animac)解释器，可以执行Scheme或JavaScript程序，未来将逐渐向能够安装应用程序的“操作系统”演进。固件代码起初为BD4SUR手工编写，后来逐渐由HI接手维护，参与编写代码的HI有云端的 Kimi K3、本地部署的 DeepSeek-V4.1-Flash、GLM5.3-Flash 等。\n\n# 功能说明\n\n番茄表：基于FLIP算法实现的流体仿真沙漏，带有重力感应和可调节的瓶颈节流功能，可当作番茄表使用。\n\n鹦鹉笼：完全运行在机器内部的端侧LLM推理，支持Qwen3和BD4SUR自制的Nano系列模型。具体能推理哪些模型，取决于设备的硬件资源。\n\n玲珑仪：是一款星图和天空仿真程序，基于星历算法，计算太阳、月亮等天体的位置，绘制在仿真的天空上。由于计算量大，在ESP32上性能不佳，但至少能实时呈现。\n\n灵机计算引擎：BD4SUR自研的Scheme和JavaScript语言解释器，提供了文字终端和软键盘，可直接在电子核桃上输入代码并执行。\n\n寻呼机：基于声学OFDM(正交频分复用调制)的声波通信功能，在两个电子核桃之间通过声波传输短消息。\n\n", color_reset_tag, color_reset_tag);

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
            ui_app_linglong_init(key_event, global_state);
            global_state->PREV_STATE = global_state->STATE;
        }

        ui_app_linglong_event_handler(key_event, global_state);
        ui_app_linglong_render_frame(key_event, global_state);

        break;

    /////////////////////////////////////////////
    // Bad Apple! 动画
    /////////////////////////////////////////////

    case STATE_BADAPPLE:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            global_state->ba_begin_timestamp = global_state->timestamp;
            global_state->ba_frame_count = 0;
        }
        global_state->PREV_STATE = global_state->STATE;

#ifdef BADAPPLE_ENABLED
        ui_app_badapple_render_frame(key_event, global_state);
#else
        ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L"未启用 Bad Apple ～", 0, 0);
        ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
#endif

        // 按A键返回主菜单
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_MAIN_MENU;
        }

        break;


    /////////////////////////////////////////////
    // FLIP流体模拟
    /////////////////////////////////////////////

    case STATE_FLIP:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_app_flip_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_app_flip_render_frame(key_event, global_state);
        ui_app_flip_event_handler(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 黄金矿工
    /////////////////////////////////////////////

    case STATE_GOLDMINER: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_goldminer_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按键处理（A返回 / D或2放钩）；返回主菜单则跳过一帧渲染
        ui_goldminer_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_GOLDMINER) break;

        // 逻辑更新 + 渲染一帧
        ui_goldminer_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 粒子生命
    /////////////////////////////////////////////

    case STATE_PARTICLELIFE: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_particlelife_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按键处理（A返回 / D或2重置）；返回则释放粒子数组并跳过一帧渲染
        ui_particlelife_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_PARTICLELIFE) {
            ui_particlelife_on_exit();
            break;
        }

        // 逻辑更新 + 渲染一帧
        ui_particlelife_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 水波
    /////////////////////////////////////////////

    case STATE_RIPPLE: {

        // 首次获得焦点：初始化（从SD卡读取/wp.png并解码为纹理）
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

        // 按键处理（A返回列表）；返回则销毁解释器并跳过一帧渲染
        ui_scheme_app_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_SCHEME_APP_RUN) {
            ui_scheme_app_on_exit();
            break;
        }

        // 泵一个时间片（解释器内部抢占分帧）；推屏时机由 App 源码控制
        ui_scheme_app_render_frame(key_event, global_state);

        // 退出 / 切后台判定，放在本帧泵完**之后**——这样 (App.exit) 返回 #t 之后，
        // App 还有整整一个时间片把当前这一拍里的收尾代码（落盘/清理）跑完，然后本帧末
        // 才销毁（退出）或冻结（切后台）。
        if (ui_scheme_app_should_exit(global_state)) {
            ui_scheme_app_on_exit();
            global_state->STATE = STATE_SCHEME_APP_LIST;
        }
        else if (ui_scheme_app_should_hide(global_state)) {
            // (App.hide) / 退出确认框的「后台」：实例与内存池保留，回列表（带 ● 角标）
            ui_scheme_app_suspend_to_list();
            global_state->STATE = STATE_SCHEME_APP_LIST;
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
    // 俄罗斯方块
    /////////////////////////////////////////////

    case STATE_TETRIS: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_tetris_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按键处理（A返回 / 移动旋转落底）；返回菜单则跳过一帧渲染
        ui_tetris_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_TETRIS) break;

        // 重力更新 + 渲染一帧
        ui_tetris_render_frame(key_event, global_state);

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
    // 计步器（时域峰值计数 + 频域周期性校验）
    /////////////////////////////////////////////

    case STATE_PEDOMETER: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_pedometer_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按A键返回主菜单
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            ui_pedometer_deinit(key_event, global_state);
            global_state->STATE = STATE_MAIN_MENU;
            break;
        }

        // 采样/分析/渲染一帧
        ui_pedometer_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 音频频谱仪：STFT声谱图（从下往上滚动）
    /////////////////////////////////////////////

    case STATE_SPECTROGRAM: {

        // 首次获得焦点：初始化（构建STFT工作区并接管麦克风）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_spectrogram_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按A键返回主菜单（关闭麦克风并恢复扬声器）
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            ui_spectrogram_deinit(key_event, global_state);
            global_state->STATE = STATE_MAIN_MENU;
            break;
        }

        // 采集一帧音频并渲染一帧声谱图
        ui_spectrogram_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 寻呼机（OFDM 声波数传）：模式菜单（发射/接收，同一时刻仅一种模式）
    /////////////////////////////////////////////

    case STATE_OFDM_MENU:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            // 统一为“页眉页脚先入帧缓冲、菜单绘制最后统一刷屏”，避免两阶段断续感
            ui_draw_header(key_event, global_state, (wchar_t *)global_state->w_menu_main->title, 1);
            ui_widget_menu_refresh(key_event, global_state, global_state->w_menu_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_widget_menu_event_handler(key_event, global_state, global_state->w_menu_main, ui_ofdm_menu_item_action, STATE_MAIN_MENU, STATE_OFDM_MENU);

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
    // 元胞自动机：Conway的生命游戏
    /////////////////////////////////////////////

    case STATE_GAMEOFLIFE:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_app_gol_init(key_event, global_state,
                (int32_t)global_state->gfx->width / UI_APP_GOL_CELL_PX, (int32_t)global_state->gfx->height / UI_APP_GOL_CELL_PX);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_app_gol_render_frame(key_event, global_state);

        // 按A键返回主菜单
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_MAIN_MENU;
        }
        // 按D键刷新
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
            ui_app_gol_init(key_event, global_state,
                (int32_t)global_state->gfx->width / UI_APP_GOL_CELL_PX, (int32_t)global_state->gfx->height / UI_APP_GOL_CELL_PX);
        }

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

        // 按A键返回主菜单
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_MAIN_MENU;
        }
        // 按D键刷新
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
            ui_app_genetic_init(key_event, global_state);
        }

        break;


    /////////////////////////////////////////////
    // 演化算法+TSP
    /////////////////////////////////////////////

    case STATE_GENETIC_TSP:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_app_tsp_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_app_tsp_refresh(key_event, global_state);

        // 按A键返回主菜单
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_MAIN_MENU;
        }
        // 按D键刷新
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
            ui_app_tsp_init(key_event, global_state);
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

        // ---- 实体键盘（十六宫格软键已整体抑制，此处仅实体键；仍显式过滤软按键）----
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

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_app_setting_value_input_draw(key_event, global_state, value_type, value_text, cursor_pos);
            gfx_refresh(global_state->gfx);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_app_setting_value_input_event_handler(key_event, global_state, value_type);

        break;


    /////////////////////////////////////////////
    // 时光集：相册
    /////////////////////////////////////////////

    case STATE_ALBUM:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {

            s_album_index = 0; // 进入相册时重置浏览索引，防止越出本次文件数范围

            gfx_soft_clear(global_state->gfx);
            gfx_draw_textline_centered(global_state->gfx, L"枚举图片文件", 160, 10, 0x66, 0xcc, 0xff, 1);

            const char *path = PLATFORM_ROOT_DIR "/image";

            // 获取数量
            int32_t count = list_files(path, NULL);
            if (count < 0) {
                printf("打开目录失败\n");
                break;
            }
            if (count == 0) {
                printf("目录为空\n");
                break;
            }

            // 分配指针数组
            s_album_path_list = (char **)platform_malloc(count * sizeof(char *));
            if (!s_album_path_list) {
                printf("内存不足\n");
                break;
            }

            // 获取文件名
            int32_t actual = list_files(path, s_album_path_list);
            if (actual < 0) {
                printf("读取失败\n");
                free(s_album_path_list);
                break;
            }
            s_album_count = actual;

            // 对文件名列表进行升序排序
            sort_strings(s_album_path_list, actual, 0);

            // 拼接成完整路径
            for (int32_t i = 0; i < actual; i++) {
                const char *filename = s_album_path_list[i];
                if (filename == NULL) continue;

                size_t path_len = strlen(path);
                size_t name_len = strlen(filename);
                int need_sep = (path_len == 0 || path[path_len - 1] == '/') ? 0 : 1;

                char *full_path = (char *)platform_malloc(path_len + need_sep + name_len + 1);
                if (full_path == NULL) {
                    printf("拼接路径内存不足\n");
                    continue;
                }

                snprintf(full_path, path_len + need_sep + name_len + 1,
                         "%s%s%s", path, need_sep ? "/" : "", filename);

                free(s_album_path_list[i]);
                s_album_path_list[i] = full_path;
            }

            // 显示文件列表
            printf("目录 %s 中有 %d 个文件:\n", path, (int)actual);
            for (int32_t i = 0; i < actual; i++) {
                wchar_t namew[128];
                _mbstowcs(namew, s_album_path_list[i], 128);
                gfx_draw_textline_centered(global_state->gfx, namew, 160, 10 + (i+1) * 17, 0xff, 0xff, 0xff, 1);
                printf("  [%d] %s\n", (int)i, s_album_path_list[i]);
            }


            gfx_refresh(global_state->gfx);
            sleep_in_ms(3000);









            gfx_draw_busy(global_state->gfx);
            gfx_refresh(global_state->gfx);
            ui_draw_image(key_event, global_state, s_album_path_list[s_album_index % s_album_count], 1);
            gfx_refresh(global_state->gfx);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 长短按1键：自动放映
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_1) {
            s_album_is_autoplay++;
            s_album_is_autoplay = s_album_is_autoplay % 2;
            s_album_refresh_timestamp = global_state->timestamp;

            ui_draw_image(key_event, global_state, s_album_path_list[s_album_index], 1);
            // if (s_album_is_autoplay) {
            //     gfx_draw_textline(global_state->gfx, L"★", 0, 0, 0x00, 0xff, 0x00, 1);
            // }
            gfx_refresh(global_state->gfx);
        }
        // 长短按2/4/5/7/8/*/0键，切换上一张图片
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && (
            (key_event->key_code == NANO_KEY_2) ||
            (key_event->key_code == NANO_KEY_4) || (key_event->key_code == NANO_KEY_5) ||
            (key_event->key_code == NANO_KEY_7) || (key_event->key_code == NANO_KEY_8) ||
            (key_event->key_code == NANO_KEY_left) || (key_event->key_code == NANO_KEY_0)
        )) {
            ui_draw_image(key_event, global_state, s_album_path_list[s_album_index], 1);
            gfx_refresh(global_state->gfx);
            s_album_index--;
            if (s_album_index < 0) s_album_index = s_album_count - 1;
            s_album_index = s_album_index % s_album_count;
        }
        // 长短按A键返回主菜单
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_MAIN_MENU;
            // 释放内存
            for (int32_t i = 0; i < s_album_count; i++) {
                printf("Freeing %s\n", s_album_path_list[i]);
                free(s_album_path_list[i]);     // 释放每个文件名
            }
            free(s_album_path_list);            // 释放指针数组
            s_album_path_list = NULL;
            // 释放图像文件缓冲区与RGB888像素缓冲区（225KB+），避免常驻PSRAM造成堆碎片
            if (s_image_file_buffer != NULL) {
                free(s_image_file_buffer);
                s_image_file_buffer = NULL;
            }
            if (s_image_rgb888_buffer != NULL) {
                free(s_image_rgb888_buffer);
                s_image_rgb888_buffer = NULL;
            }
            s_image_file_size = 0;
            s_image_decode_ready = 0;
            s_image_filename_buffer[0] = '\0'; // 使下次进入相册时重新读文件解码
            printf("Free done.\n");
        }
        // 长短按3/6/B/9/C/#/D键，切换下一张图片
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && (
            (key_event->key_code == NANO_KEY_3) ||
            (key_event->key_code == NANO_KEY_6) || (key_event->key_code == NANO_KEY_shift) ||
            (key_event->key_code == NANO_KEY_9) || (key_event->key_code == NANO_KEY_ctrl) ||
            (key_event->key_code == NANO_KEY_right) || (key_event->key_code == NANO_KEY_enter)
        )) {
            ui_draw_image(key_event, global_state, s_album_path_list[s_album_index], 1);
            gfx_refresh(global_state->gfx);
            s_album_index++;
            s_album_index = s_album_index % s_album_count;
        }

        // 自动放映
        if (s_album_is_autoplay) {
            // 每6000ms切换
            if (global_state->timestamp - s_album_refresh_timestamp >= 6000) {
                ui_draw_image(key_event, global_state, s_album_path_list[s_album_index], 1);
                // gfx_draw_textline(global_state->gfx, L"★", 0, 0, 0x00, 0xff, 0x00, 1);
                gfx_refresh(global_state->gfx);
                s_album_index++;
                s_album_index = s_album_index % s_album_count;
                s_album_refresh_timestamp = global_state->timestamp;
            }
            else {
                // int32_t aa = (float)(global_state->timestamp - s_album_refresh_timestamp) / 6000.0f * 360;
                // gfx_draw_circle_fill(global_state->gfx, 6, 6, 6, 0xff, 0xff, 0xff, 1);
                // gfx_draw_sector(global_state->gfx, 6, 6, 6, 0, aa, 0x66, 0xcc, 0xff, 1);

                int32_t w = (float)(global_state->timestamp - s_album_refresh_timestamp) / 6000.0f * global_state->gfx->width;
                // gfx_draw_rectangle(global_state->gfx, 0, 239, global_state->gfx->width, 1, 0x00, 0x00, 0x00, 1);
                gfx_draw_rectangle(global_state->gfx, 0, 239, w, 1, 0x00, 0xaa, 0xff, 1);
                gfx_refresh(global_state->gfx);
            }
        }

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

        // 首次获得焦点：叠加绘制模态框（控制台画面保留在其下）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_exit_confirm_draw(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 模态期间不泵编辑器事件循环（ui_animac_idle_pump 暂停）：
        // 避免异步输出触发整帧重绘覆盖模态框；运行时定时器随之暂停，对话框存续期间可接受

        // 触屏按钮（松手沿 + 按下点命中，全局范式）
        if (key_event->touch_edge & TOUCH_EDGE_UP) {
            int32_t hit = ui_exit_confirm_hit(global_state, key_event->touch_down_x, key_event->touch_down_y);
            if (hit == 1) {
                // 确认退出：完整善后（解释器内存/字体/终端布局关联/日志区几何）后回主菜单
                ui_app_animac_cleanup(key_event, global_state);
                global_state->STATE = STATE_MAIN_MENU;
            }
            else if (hit == 2) {
                // 留下：恢复控制台画面（整帧重绘，终端模式同帧合成日志区）并返回
                ui_draw_input_buffer(key_event, global_state, global_state->w_input_main);
                global_state->STATE = STATE_ANIMAC_CONSOLE;
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