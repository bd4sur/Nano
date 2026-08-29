// hal_misc M5Core2 / CoreS3 实现：指示灯 / 振动马达 / 蜂鸣
//（机型差异经 platform.h 的 NANO_PLATFORM_* 宏分支）

#include <M5Unified.h>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include "platform.h"
#include "hal_os.h"
#include "hal_misc.h"

// 跨核互斥量（程序初始化期、任务创建前的单线程环境静态创建，无初始化竞态）：
//  - s_led_mutex：LED 驱动（CoreS3 RMT 灯带 / Core2 PMIC I2C）非跨核线程安全，
//    双核按键反馈与 misc_led_poll 的高频并发访问曾致驱动层死锁卡死（2026-08 实测）；
//  - s_spk_mutex：M5Unified 扬声器 begin/spk_task/I2S 通道建立非线程安全，
//    双核按键反馈近乎同时调 tone 曾致 CoreS3 spk_task 空指针+栈金丝雀崩溃（2026-08 实测）。
static SemaphoreHandle_t s_led_mutex = xSemaphoreCreateMutex();
static SemaphoreHandle_t s_spk_mutex = xSemaphoreCreateMutex();

// ---------------- 指示灯 ----------------

#if defined(NANO_PLATFORM_M5CORES3)

#include <memory>

// M5Unified LED 灯带支持（RMT 总线 + WS2812 灯带），参照 M5Unified led_class 文档
#include "utility/led/LED_Strip_Class.hpp"

// M5GO3 Bottom 底座灯带：10 颗 WS2812（每侧 5 颗），数据线位于 M-Bus pin8，
// 对 CoreS3 即 GPIO5（见 M5GO3 Bottom 原理图 M-Bus 连接器 J4 pin8 网络标号 RGB → GPIO5）
#define LED_STRIP_PIN_DATA   (5)
#define LED_STRIP_COUNT      (10)

void misc_led_init(void) {
    // 使能 M-Bus 5V 输出（AW9523 BUS_EN + SY7088 BOOST_EN），底座灯带由此取电
    M5.Power.setExtOutput(true);

    // 注册灯带实例到 M5.Led（RMT 总线 + GRB 灯带）
    auto bus = std::make_shared<m5::LedBus_RMT>();
    auto bus_cfg = bus->getConfig();
    bus_cfg.pin_data = LED_STRIP_PIN_DATA;
    bus->setConfig(bus_cfg);

    auto strip = std::make_shared<m5::LED_Strip_Class>();
    auto strip_cfg = strip->getConfig();
    strip_cfg.led_count = LED_STRIP_COUNT;
    strip_cfg.byte_per_led = 3;
    strip_cfg.color_order = m5::LED_Strip_Class::config_t::color_order_grb;
    strip->setConfig(strip_cfg);
    strip->setBus(bus);

    M5.Led.setLedInstance(strip);
    M5.Led.setBrightness(255);      // 亮度调至最亮
    M5.Led.setAutoDisplay(false);   // 改色后统一 display，避免逐颗推帧
    M5.Led.begin();
    misc_led_set(0, MISC_LED_COLOR_BLUE);
}

static void misc_led_set_raw(int32_t on, int32_t color) {
    if (on) {
        if (color == MISC_LED_COLOR_GREEN) M5.Led.setAllColor(0, 255, 0);
        else                               M5.Led.setAllColor(0, 0, 255);
    }
    else {
        M5.Led.setAllColor(0, 0, 0);
    }
    M5.Led.display();
}

#else // NANO_PLATFORM_M5CORE2

void misc_led_init(void) {
    // Core2 自带 LED 由 PMIC 控制（M5Unified Power_Class::setLed 内部按 AXP192/AXP2101 自适应）
    M5.Power.setLed(0);
}

static void misc_led_set_raw(int32_t on, int32_t color) {
    (void)color; // 自带 LED 为单色，颜色参数忽略
    M5.Power.setLed(on ? 188 : 0);
}

#endif

// 指示灯亮/灭（跨核互斥入口；驱动非线程安全，见文件头部互斥量注释）
void misc_led_set(int32_t on, int32_t color) {
    if (s_led_mutex != NULL) { xSemaphoreTake(s_led_mutex, portMAX_DELAY); }
    misc_led_set_raw(on, color);
    if (s_led_mutex != NULL) { xSemaphoreGive(s_led_mutex); }
}

// 指示灯闪烁一次（异步非阻塞：立即点亮，由 misc_led_poll 按截止时间熄灭）
// 32位毫秒时间戳 + 回绕安全比较；跨核调用仅需 32 位原子写（Xtensa 对齐 32 位访问原子）
static volatile uint32_t s_led_off_at_ms = 0; // 熄灭截止时间戳（ms 低 32 位）
static volatile uint8_t  s_led_off_armed = 0; // 1-有点亮待熄灭

void misc_led_blink(int32_t color, uint32_t duration_ms) {
    misc_led_set(1, color);
    s_led_off_at_ms = (uint32_t)get_timestamp_in_ms() + duration_ms;
    s_led_off_armed = 1;
}

void misc_led_poll(void) {
    if (s_led_off_armed && (int32_t)((uint32_t)get_timestamp_in_ms() - s_led_off_at_ms) >= 0) {
        s_led_off_armed = 0;
        misc_led_set(0, MISC_LED_COLOR_BLUE); // 熄灭与颜色无关（Core2 单色；CoreS3 整带灭）
    }
}

// ---------------- 振动马达 ----------------
// 振动(0-255)
void set_vibration(uint32_t level) {
    M5.Power.setVibration(level);
}

// ---------------- 蜂鸣（扬声器 tone 提示音） ----------------

// M5Unified 扬声器（spk_task 与 I2S 通道的建立/拆除）非线程安全：
// 双核按键反馈（Core1 即时侧 4000Hz + Core0 队列侧 6000Hz）会对同一按键事件近乎同时调用
// M5.Speaker.tone → _play_raw → begin()；两个 begin() 并发执行 _setup_i2s（先 uninstall 置空句柄
// 再 new_channel）并重复创建 spk_task，I2S 通道句柄被对侧删除/复用后空指针解引用
//（2026-08 CoreS3 实测 Guru Meditation：spk_task → i2s_channel_enable → i2s_tx_channel_start
//  LoadProhibited(EXCVADDR=0) + spk_task 栈金丝雀）。故对 tone 入口做跨核互斥。
// 互斥量在程序初始化期（任务创建前、单线程环境）静态创建，无初始化竞态（声明见文件头部）。

void misc_tone(uint32_t freq_hz, uint32_t duration_ms) {
    if (s_spk_mutex != NULL) { xSemaphoreTake(s_spk_mutex, portMAX_DELAY); }
    M5.Speaker.tone(freq_hz, duration_ms);
    if (s_spk_mutex != NULL) { xSemaphoreGive(s_spk_mutex); }
}
