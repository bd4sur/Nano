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

        if (xQueueReceive(event_queue, &key_event_0, 0) != pdTRUE) {
            // Serial.println("No event received");
            key_event_0.key_code = NANO_KEY_IDLE;
            key_event_0.key_edge = 0;
        }

        if (key_event_0.key_code != NANO_KEY_IDLE) {
            if (key_event_0.key_edge < 0) {
                // 按键提示-蜂鸣（Core0 队列侧，6000Hz）：OFDM 寻呼机发射/接收、音乐盒播放、声谱图状态下禁用按键音
                //（避免抢占麦克风 I2S、污染发射信号、抢占扬声器通道干扰音乐；
                //  声谱图：tone 经 Speaker 争用 I2S 会导致麦克风采集链路中断、声谱图消失）
                if ((global_state->key_feedback_mode & 2) &&
                    global_state->STATE != STATE_OFDM_TXING && global_state->STATE != STATE_OFDM_RX &&
                    global_state->STATE != STATE_MUSICBOX_PLAYING &&
                    global_state->STATE != STATE_SPECTROGRAM) {
                    misc_tone(6000, 10);
                }
                // 按键提示-灯光（Core0 队列侧：绿色，同步阻塞点亮 100ms）
                if (global_state->key_feedback_mode & 1) {
                    misc_led_blink(MISC_LED_COLOR_GREEN, KEY_LED_ON_DURATION_MS);
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
    event_queue = xQueueCreate(2, sizeof(Key_Event));


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

    // 物理时间戳
    global_state->timestamp = get_timestamp_in_ms();

    // 获取按键事件
    get_key_event(&key_event_1, global_state);
    if (key_event_1.key_code != NANO_KEY_IDLE && key_event_1.key_edge < 0) {
        // 按键提示-蜂鸣（Core1 即时侧，4000Hz）：OFDM 寻呼机发射/接收、音乐盒播放、声谱图状态下禁用按键音
        if ((global_state->key_feedback_mode & 2) &&
            global_state->STATE != STATE_OFDM_TXING && global_state->STATE != STATE_OFDM_RX &&
            global_state->STATE != STATE_MUSICBOX_PLAYING &&
            global_state->STATE != STATE_SPECTROGRAM) {
            misc_tone(4000, 10);
        }
        // 按键提示-灯光（Core1 即时侧：蓝色，同步阻塞点亮 100ms；Core0 队列侧为绿色）：
        // 无 I2S 争用问题，所有状态下均生效
        if (global_state->key_feedback_mode & 1) {
            misc_led_blink(MISC_LED_COLOR_BLUE, KEY_LED_ON_DURATION_MS);
        }
    }

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

        // 仅在下降沿发送事件：NOTE 假设业务逻辑只认下降沿
        if (key_event_1.key_code != NANO_KEY_IDLE && key_event_1.key_edge < 0) {
            // Serial.println("Send");
            // Serial.println(key_event_1.key_code);
            // Serial.println(key_event_1.key_edge);
            // 仅玲珑仪显示提示
            if (global_state->STATE == STATE_LINGLONG || global_state->STATE == STATE_ALBUM) {
                gfx_draw_busy(global_state->gfx);
                gfx_refresh(global_state->gfx);
            }
            if (xQueueSend(event_queue, &key_event_1, pdMS_TO_TICKS(1)) != pdTRUE) {
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

    vTaskDelay(1);
}