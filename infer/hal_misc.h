#ifndef __NANO_HAL_MISC_H__
#define __NANO_HAL_MISC_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===============================================================================
// hal_misc：杂项外设抽象层（硬件无关接口；ESP32 实现见 hal_misc_m5esp.cpp）
//   指示灯（按键反馈灯光）/ 振动马达 / 蜂鸣（扬声器 tone 提示音）
// ===============================================================================

// ---------------- 指示灯 ----------------
//   Core2  ：机器自带 LED（PMIC 控制，单色；颜色参数忽略）
//   CoreS3 ：M5GO3 Bottom 底座灯带（10 颗 WS2812 整带同色；数据线位于 M-Bus pin8，
//            对 CoreS3 即 GPIO5——见 M5GO3 Bottom 原理图；灯带 5V 供电取自 M-Bus，
//            需经 AW9523 使能 BUS_EN/BOOST）
#define MISC_LED_COLOR_BLUE   (0)
#define MISC_LED_COLOR_GREEN  (1)
#define MISC_LED_COLOR_RED    (2)
#define MISC_LED_COLOR_YELLOW (3)
#define MISC_LED_COLOR_CYAN   (4)
#define MISC_LED_COLOR_PURPLE (5)
#define MISC_LED_COLOR_WHITE  (6)

// 初始化（在 M5.begin 之后调用一次）
void misc_led_init(void);

// 指示灯亮/灭（on：1=亮 0=灭；color：MISC_LED_COLOR_*，仅 CoreS3 灯带有效；
// brightness：亮度 0~255，仅支持的机型生效（Core2 PMIC PWM / CoreS3 灯带均支持），
// 不支持的机型忽略；非阻塞，熄灭时机由调用方计时控制）
void misc_led_set(int32_t on, int32_t color, uint8_t brightness);

// 指示灯闪烁一次（异步非阻塞：立即点亮，由 misc_led_poll 在到达点亮时长后熄灭；
// brightness 语义同 misc_led_set）
void misc_led_blink(int32_t color, uint8_t brightness, uint32_t duration_ms);
// 指示灯异步熄灭的轮询推进：需在双核各自的主循环中高频调用（.ino loop 与 core0_render_task）
void misc_led_poll(void);

// ---------------- 振动马达 ----------------
// 振动(0-255)
void set_vibration(uint32_t level);

// ---------------- 蜂鸣（扬声器 tone 提示音） ----------------
// 非阻塞提示音（freq_hz：频率 Hz；duration_ms：时长 ms）
void misc_tone(uint32_t freq_hz, uint32_t duration_ms);

#ifdef __cplusplus
}
#endif

#endif
