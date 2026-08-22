// hal_misc M5Core2 / CoreS3 实现：指示灯 / 振动马达 / 蜂鸣
//（机型差异经 platform.h 的 NANO_PLATFORM_* 宏分支）

#include <M5Unified.h>

#include "platform.h"
#include "hal_misc.h"

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

void misc_led_set(int32_t on, int32_t color) {
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

void misc_led_set(int32_t on, int32_t color) {
    (void)color; // 自带 LED 为单色，颜色参数忽略
    M5.Power.setLed(on ? 255 : 0);
}

#endif

// 指示灯闪烁一次（同步阻塞：点亮 → 延时 → 熄灭）
void misc_led_blink(int32_t color, uint32_t duration_ms) {
    misc_led_set(1, color);
    delay(duration_ms);
    misc_led_set(0, color);
}

// ---------------- 振动马达 ----------------
// 振动(0-255)
void set_vibration(uint32_t level) {
    M5.Power.setVibration(level);
}

// ---------------- 蜂鸣（扬声器 tone 提示音） ----------------

void misc_tone(uint32_t freq_hz, uint32_t duration_ms) {
    M5.Speaker.tone(freq_hz, duration_ms);
}
