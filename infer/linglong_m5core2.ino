#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "esp_task_wdt.h"
#include "esp_heap_caps.h"

#include <M5Unified.h>
#include <M5GFX.h>

#include "hal_key.h"
#include "hal_audio_out.h"
#include "hal_misc.h"
#include "hal_power.h"
#include "ui_app.h"
#include "platform.h"
#include "celestial.h"
#include "nongli.h"

// M5GFX display;

static Global_State *global_state = NULL;
static Key_Event     key_event_0 = {0};
static Key_Event     key_event_1 = {0};
Nano_GFX *gfx;

#define UI_STATE_DEFAULT (0)
#define UI_STATE_SKY     (1)
#define UI_STATE_SETTING (2)
#define UI_STATE_README  (3)


static TaskHandle_t core0_task_handle = NULL;

// 按键提示灯光点亮时长（ms，同步阻塞）
#define KEY_LED_ON_DURATION_MS (10)

// 锁屏激活判定：无 PMIC 电源键平台（NANO_HAS_PWR_KEY=0）恒假，锁屏相关抑制逻辑不参与
#if NANO_HAS_PWR_KEY
#define LOCK_SCREEN_ACTIVE(gs) ((gs)->STATE == STATE_LOCK_SCREEN)
#else
#define LOCK_SCREEN_ACTIVE(gs) (0)
#endif

// Core0 → Core1: 帧就绪通知
static QueueHandle_t frame_ready_queue = NULL;
// Core1 → Core0: 帧消费确认
static QueueHandle_t frame_consumed_queue = NULL;
// Core1 → Core0
static QueueHandle_t event_queue = NULL;



// 模拟 timegm：struct tm (UTC) → time_t
time_t esp_timegm(struct tm *tm) {
    time_t t;
    char *tz = getenv("TZ");

    setenv("TZ", "UTC0", 1);
    tzset();
    t = mktime(tm);

    if (tz) setenv("TZ", tz, 1);
    else     unsetenv("TZ");
    tzset();

    return t;
}

void core0_render_task(void *pvParameters) {

    // 将当前任务注册到看门狗
    // esp_err_t err = esp_task_wdt_add(NULL); // NULL = 当前任务
    // if (err != ESP_OK) {
    //     Serial.printf("WDT add failed: %d\n", err);
    //     vTaskDelete(NULL);
    // }

    uint8_t dummy = 1;

    while (1) {
        // esp_task_wdt_reset();

        misc_led_poll(); // 指示灯异步熄灭推进（按键反馈灯光已异步化）

        // 每帧排空事件队列（触屏事件队列改造，见 AGENTS.md 第八节）：
        //  - 触屏 DOWN/UP 边沿事件（key_code==NANO_KEY_IDLE 且 touch_edge!=0）：
        //    位掩码并入本帧（同帧 DOWN+UP 叠加为 3，亚帧短点按不湮灭），坐标/电平/
        //    按下点取事件内最新值，继续排空；
        //  - 按键事件：维持“每帧一个”语义——取出首个按键事件即停止排空，其余留队列下帧处理。
        key_event_0.touch_edge = 0;
        uint8_t frame_touch_edge = 0;
        {
            Key_Event ev;
            int32_t got_key = 0;
            while (xQueueReceive(event_queue, &ev, 0) == pdTRUE) {
                if (ev.key_code == NANO_KEY_IDLE && ev.touch_edge != 0) {
                    frame_touch_edge |= ev.touch_edge;
                    key_event_0.touch_x = ev.touch_x;
                    key_event_0.touch_y = ev.touch_y;
                    key_event_0.is_touching = ev.is_touching;
                    key_event_0.touch_down_x = ev.touch_down_x;
                    key_event_0.touch_down_y = ev.touch_down_y;
                    continue;
                }
                key_event_0 = ev; // 按键事件：整体沿用（其触屏字段亦为生产端最新值）
                got_key = 1;
                break;
            }
            if (!got_key) {
                key_event_0.key_code = NANO_KEY_IDLE;
                key_event_0.key_edge = 0;
            }
        }
        key_event_0.touch_edge = frame_touch_edge;

        // 触屏电平/坐标取自 Core1 在 get_input_event 中高频锁存的共享快照；
        // 业务逻辑统一经 key_event 消费触屏，不跨层直读触屏HAL；
        // 触屏 DOWN/UP 边沿事件已经上方队列排空合入 touch_edge，
        // 此处仅刷新轨迹电平（touch_edge/touch_down_* 不被覆盖）
        key_event_0.touch_x = global_state->touch_x;
        key_event_0.touch_y = global_state->touch_y;
        key_event_0.is_touching = global_state->is_touching;

        if (key_event_0.key_code != NANO_KEY_IDLE) {
            if (key_event_0.key_edge == -1) { // 反馈仅认短按下降沿：-2 长按重复事件流（1kHz）不做反馈
                // 按键提示-蜂鸣（Core0 队列侧，6000Hz）：OFDM 寻呼机发射/接收、音乐盒播放、声谱图状态下禁用按键音
                //（避免抢占麦克风 I2S、污染发射信号、抢占扬声器通道干扰音乐；
                //  声谱图：tone 经 Speaker 争用 I2S 会导致麦克风采集链路中断、声谱图消失）
                if ((global_state->key_feedback_mode & 2) &&
                    global_state->STATE != STATE_OFDM_TXING && global_state->STATE != STATE_OFDM_RX &&
                    global_state->STATE != STATE_MUSICBOX_PLAYING &&
                    global_state->STATE != STATE_SPECTROGRAM &&
                    !LOCK_SCREEN_ACTIVE(global_state)) {
                    misc_tone(6000, 10);
                }
                // 按键提示-灯光（Core0 队列侧：绿色，异步点亮，misc_led_poll 到时熄灭）
                // 锁屏态禁用：与锁屏 LED 心跳互斥
                if ((global_state->key_feedback_mode & 1) &&
                    !LOCK_SCREEN_ACTIVE(global_state)) {
                    misc_led_blink(MISC_LED_COLOR_GREEN, 128, KEY_LED_ON_DURATION_MS);
                }
            }
            // Serial.println("Receive");
            // Serial.println(key_event_0.key_code);
            // Serial.println(key_event_0.key_edge);

            // 仅玲珑仪显示提示
            if (global_state->STATE == STATE_LINGLONG || global_state->STATE == STATE_ALBUM) {
                gfx_draw_busy(global_state->gfx);
                gfx_refresh(global_state->gfx);
            }
        }


        // 事件处理器
        main_event_handler(&key_event_0, global_state);
        // 周期性任务
        main_periodic_task(&key_event_0, global_state);


        // 1. 通知 Core1 帧已就绪（阻塞直到发送成功，形成背压）
        // if (xQueueSend(frame_ready_queue, &dummy, pdMS_TO_TICKS(1000)) != pdTRUE) {
        //     Serial.println("frame_ready_queue send timeout!");
        //     continue; // 跳过当前帧，避免堆积
        // }

        // 2. 等待 Core1 消费确认（带超时保护）
        // if (xQueueReceive(frame_consumed_queue, &dummy, pdMS_TO_TICKS(2000)) != pdTRUE) {
        //     Serial.println("frame_consumed_queue timeout!");
        // }

        // 让出时间片（避免独占Core0）
        vTaskDelay(0);
    }
}





void setup() {
    Serial.begin(115200);

    esp_task_wdt_deinit();

    // 尽早分配帧缓冲区（双缓冲，适配DMA最大连续块）：
    // 两块76.8KB需位于DMA可寻址的内部RAM，必须在堆最完整、未碎片化时分配，
    // 否则M5.begin/gfx_init等初始化消耗并切碎内部堆后，很可能找不到足够的连续块
    uint16_t *frame_buffer_top = (uint16_t *)heap_caps_malloc(
        SCREEN_WIDTH * (SCREEN_HEIGHT / 2) * sizeof(uint16_t), MALLOC_CAP_DMA);
    uint16_t *frame_buffer_bottom = (uint16_t *)heap_caps_malloc(
        SCREEN_WIDTH * (SCREEN_HEIGHT / 2) * sizeof(uint16_t), MALLOC_CAP_DMA);

    if (!frame_buffer_top || !frame_buffer_bottom) {
        Serial.println("Failed to alloc frame buffers!");
        while (1) delay(1000);
    }

    auto cfg = M5.config();
    M5.begin(cfg);

#if NANO_HAS_PWR_KEY
    // 丢弃上电残留的电源键（PMIC PEK）锁存状态，避免开机即误触发锁屏
    power_key_poll();
#endif

    //////////////////////////////////////////////////
    // 查看内存用量
    //////////////////////////////////////////////////

    Serial.printf("DRAM Free: %u bytes\n", 
                heap_caps_get_free_size(MALLOC_CAP_8BIT));  // DRAM 堆
    Serial.printf("Largest Block: %u bytes\n", 
                    heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    Serial.printf("PSRAM Free: %u bytes\n", 
                    heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    Serial.printf("DMA-capable Free: %u bytes | Largest Block: %u bytes\n",
                    heap_caps_get_free_size(MALLOC_CAP_DMA),
                    heap_caps_get_largest_free_block(MALLOC_CAP_DMA));

    global_state = (Global_State*)platform_calloc(1, sizeof(Global_State));

    //////////////////////////////////////////////////
    // 设置GFX
    //////////////////////////////////////////////////

    global_state->gfx = (Nano_GFX*)platform_calloc(1, sizeof(Nano_GFX));
    global_state->gfx->is_double_buffer = 1;
    gfx_init(global_state->gfx, SCREEN_WIDTH, SCREEN_HEIGHT, GFX_COLOR_MODE_RGB565);

    // 挂上已在 setup 开头分配好的DMA帧缓冲区
    global_state->gfx->frame_buffer_rgb565_top = frame_buffer_top;
    global_state->gfx->frame_buffer_rgb565_bottom = frame_buffer_bottom;

    delay(100);

    memset(global_state->gfx->frame_buffer_rgb565_top, 0, SCREEN_WIDTH * (SCREEN_HEIGHT / 2) * sizeof(uint16_t));
    memset(global_state->gfx->frame_buffer_rgb565_bottom, 0, SCREEN_WIDTH * (SCREEN_HEIGHT / 2) * sizeof(uint16_t));


    main_init(&key_event_1, global_state);


    // 全局主音量（ui_init 已初始化 global_state->volume；同时应用到扬声器硬件）
    audio_out_set_master_volume((uint8_t)global_state->volume);

    // 按键指示灯初始化（Core2：自带LED；CoreS3：M5GO3 Bottom 底座灯带）
    misc_led_init();


    ui_app_splash_render_frame(&key_event_1, global_state);

    setenv("TZ", "CST-8", 1);
    tzset();

    // esp_task_wdt_config_t wdt_config = {
    //     .timeout_ms = 30000,      // 30 秒
    //     .idle_core_mask = 0,
    //     .trigger_panic = true     // 超时后 panic 复位
    // };
    // esp_task_wdt_init(&wdt_config);


    // Core0 → Core1: 帧就绪通知
    frame_ready_queue = xQueueCreate(1, sizeof(uint8_t));
    // Core1 → Core0: 帧消费确认
    frame_consumed_queue = xQueueCreate(1, sizeof(uint8_t));
    event_queue = xQueueCreate(8, sizeof(Key_Event)); // 触屏边沿事件入队后为触键混合排队留余量（原长度2）


    // 创建 Core0 渲染任务（12KB栈：Animac解释器递归求值需要较大栈空间）
    xTaskCreatePinnedToCore(
        core0_render_task,
        "render",
        12280,     // 栈大小（12KB）
        NULL,
        1,         // 低优先级（避免饿死Core1的中断）
        &core0_task_handle,
        0          // 固定到 Core 0
    );

    Serial.println("Setup done");
}




void loop() {
    M5.update();

    misc_led_poll(); // 指示灯异步熄灭推进（按键反馈灯光已异步化）

    // 物理时间戳
    global_state->timestamp = get_timestamp_in_ms();

#if NANO_HAS_PWR_KEY
    // 电源键（PMIC PEK）轮询：经 M5.BtnPWR（M5.update 已在本轮顶部执行，状态机为最新）；
    // 读取为锁存标志、零 I2C 开销，无需节流。检出短按 → 构造 NANO_KEY_POWER 下降沿事件可靠投递
    //（锁屏/解锁由 Core0 全局拦截统一处理）。锁屏期间保留本轮询（解锁唯一入口），
    // 其余按键/触屏事件生产见下方抑制。注意不可旁路直读 getKeyState（读清式，会被 M5.update 抢走）。
    if (power_key_poll() == 1) {
        Key_Event pwr_ev = {0};
        pwr_ev.key_code = NANO_KEY_POWER;
        pwr_ev.key_edge = -1;
        if (xQueueSend(event_queue, &pwr_ev, pdMS_TO_TICKS(1)) != pdTRUE) {
            Serial.println("WARNING: event_queue power key send timeout!");
        }
    }
#endif

    // 获取输入事件（按键 + 触屏）
    // NOTE 按键反馈（misc_led_blink 同步阻塞 ~10ms）已移至本循环末尾、事件入队之后：
    // 若先反馈再入队，阻塞期间触屏电平快照已翻转而边沿事件滞留未发，
    // Core0 会在窗口内看到“电平0+无边沿”的中间态（2026-08 电子书短按丢失故障之根因）
    get_input_event(&key_event_1, global_state);

/*
    if (global_state->STATE == STATE_SPLASH_SCREEN && key_event_1.key_code == KEYCODE_NUM_0 && key_event_1.key_edge < 0) {
        if (eTaskGetState(core0_task_handle) != eSuspended) {
            vTaskSuspend(core0_task_handle);  // 暂停 Core0 任务
            Serial.println("core0_render_task paused");
        }

        play_badapple();

        if (eTaskGetState(core0_task_handle) == eSuspended) {
            vTaskResume(core0_task_handle);   // 恢复 Core0 任务
            Serial.println("core0_render_task resumed");
        }
    }
*/

    uint8_t dummy = 1;
    // if (xQueueReceive(frame_ready_queue, &dummy, 0) == pdTRUE) {

        // gfx_refresh(global_state->gfx);

        // 发送按键事件到 Core0（NOTE 假设业务逻辑只认下降沿），分两类投递：
        //  - 可靠类（1ms超时+告警，不可丢弃）：短按下降沿(-1)——一次性事件；
        //  - 可丢弃类（0超时、队列满静默丢弃）：长按/重复动作(-2)。get_input_event 的重复机制
        //    在按住超过360ms后以 Core1 轮询速率（~1kHz，远高于 Core0 帧率）持续产生 -2 流，
        //    若对其可靠发送，会占满队列（长度仅2），1ms超时失败后的阻塞式串口告警反过来
        //    拖垮 Core1，表现为"响应几个事件→卡住→再响应"的循环；静默丢弃后 Core0 每帧
        //    总能取到最新的重复事件，重复节奏自然对齐 Core0 帧率（与 Linux 单循环端一致）。
        // 触屏电平不走队列——由 get_input_event 高频锁存到 Global_State 共享快照
        // （touch_x/touch_y/is_touching），Core0 每帧直接取用覆盖到 key_event_0。
        // 触屏 DOWN/UP 边沿事件（触屏事件队列改造，见 AGENTS.md 第八节）：
        // 可靠投递（同短按 -1 策略：1ms 超时+告警，一次性事件不可丢）。
        // 键码字段置 NANO_KEY_IDLE 以符合“触屏事件=key_code IDLE 且 touch_edge 非0”的身份约定；
        // 移动轨迹不入队，仍走共享快照（见 get_input_event 注释）。
        // NOTE 必须先于按键事件投递：同一次松手会同时产生触屏 UP 与宫格软按键下降沿，
        // 若按键先入队，Core0 会先消费到滞后的软按键（如热点位置被映射为 D 键误提交），
        // 触屏 UP 后到时热点已无法拦下它；先投触屏事件可保证同帧内热点判定先于按键处理。
        // 锁屏期间抑制触屏边沿事件投递（防误触堆积队列；电源键事件不受此限）
        if (key_event_1.touch_edge != 0 && !LOCK_SCREEN_ACTIVE(global_state)) {
            Key_Event touch_ev = key_event_1;
            touch_ev.key_code = NANO_KEY_IDLE;
            touch_ev.key_edge = 0;
            if (xQueueSend(event_queue, &touch_ev, pdMS_TO_TICKS(1)) != pdTRUE) {
                Serial.println("WARNING: event_queue touch event send timeout!");
            }
        }

        // 锁屏期间抑制按键事件投递（同上）
        if (key_event_1.key_code != NANO_KEY_IDLE && key_event_1.key_edge < 0 &&
            !LOCK_SCREEN_ACTIVE(global_state)) {
            int32_t is_reliable = (key_event_1.key_edge == -1);
            // Serial.println("Send");
            // Serial.println(key_event_1.key_code);
            // Serial.println(key_event_1.key_edge);
            // 仅玲珑仪/时光集显示忙提示
            if (global_state->STATE == STATE_LINGLONG || global_state->STATE == STATE_ALBUM) {
                gfx_draw_busy(global_state->gfx);
                gfx_refresh(global_state->gfx);
            }
            if (xQueueSend(event_queue, &key_event_1, (is_reliable) ? pdMS_TO_TICKS(1) : 0) != pdTRUE &&
                is_reliable) {
                Serial.println("WARNING: event_queue send timeout!");
            }
        }

        // 通知 Core0 帧已消费（非阻塞）
        // if (xQueueSend(frame_consumed_queue, &dummy, 0) != pdTRUE) {
        //     Serial.println("WARNING: frame_consumed_queue full! Core0 may be stuck.");
        // }
    // }

    // 更新上一轮循环的物理时间戳
    global_state->timestamp_last = global_state->timestamp;

    // 按键提示反馈（Core1 即时侧）：置于事件入队之后，避免阻塞延迟事件投递（见上方 NOTE）
    // 反馈仅认短按下降沿（-1）：-2 长按重复事件流（1kHz）不做反馈——高频并发访问
    // LED 驱动（非跨核线程安全）曾致双核死锁卡死（2026-08 实测）
    // 锁屏期间抑制按键反馈（键盘事件本就不投递，此处为兜底）
    if (key_event_1.key_code != NANO_KEY_IDLE && key_event_1.key_edge == -1 &&
        !LOCK_SCREEN_ACTIVE(global_state)) {
        // 蜂鸣（4000Hz）：OFDM 寻呼机发射/接收、音乐盒播放、声谱图状态下禁用按键音
        if ((global_state->key_feedback_mode & 2) &&
            global_state->STATE != STATE_OFDM_TXING && global_state->STATE != STATE_OFDM_RX &&
            global_state->STATE != STATE_MUSICBOX_PLAYING &&
            global_state->STATE != STATE_SPECTROGRAM &&
            !LOCK_SCREEN_ACTIVE(global_state)) {
            misc_tone(4000, 10);
        }
        // 灯光（蓝色，异步点亮，misc_led_poll 到时熄灭；Core0 队列侧为绿色）：无 I2S 争用问题，所有状态下均生效
        if (global_state->key_feedback_mode & 1) {
            misc_led_blink(MISC_LED_COLOR_BLUE, 128, KEY_LED_ON_DURATION_MS);
        }
    }

    vTaskDelay(1);
}