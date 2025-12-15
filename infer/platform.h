#ifndef __NANO_PLATFORM_H__
#define __NANO_PLATFORM_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

// ===============================================================================
// 平台相关工具函数
// ===============================================================================

void sleep_in_ms(uint32_t ms);
uint32_t get_timestamp_in_ms();
int32_t graceful_shutdown();

// ===============================================================================
// Nano-Pod: Raspberry Pi 5
// ===============================================================================
#if defined(NANO_POD_RPI5)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    #define I2C_DEVFILE "/dev/i2c-1"
    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"
    // 屏幕
    #define SSD1309
    #define OLED_I2C_ADDR (0x3c)
    // 键盘
    #define KB_I2C_ADDR (0x27)
    // UPS
    #define UPS_ENABLED
    #define UPS_I2C_ADDR (0x36)
    // ASR和TTS
    #define ASR_ENABLED
    #define TTS_ENABLED
    #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"
    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

// ===============================================================================
// Nano-Pod: Rock 5B+
// ===============================================================================
#elif defined(NANO_POD_ROCK5BP)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    #define I2C_DEVFILE "/dev/i2c-3"
    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"
    // 屏幕
    #define SSD1309
    #define OLED_I2C_ADDR (0x3c)
    // 键盘
    #define KB_I2C_ADDR (0x27)
    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)
    // ASR和TTS
    #define ASR_ENABLED
    #define TTS_ENABLED
    #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"
    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

// ===============================================================================
// Nano-Pod: Cubie-A7Z
// ===============================================================================
#elif defined(NANO_POD_CUBIE_A7Z)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    #define I2C_DEVFILE "/dev/i2c-7"
    // 模型目录
    #define MODEL_ROOT_DIR "/home/radxa/ai/_model/Nano"
    // 屏幕
    #define SSD1306
    #define OLED_I2C_ADDR (0x3c)
    // 键盘
    #define KB_I2C_ADDR (0x27)
    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)
    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"
    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

// ===============================================================================
// Nano-Pod: Make Router Great Again 京东云RE-CS-02、红米AX5等
// ===============================================================================
#elif defined(NANO_POD_MARGA)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    #define I2C_DEVFILE "/dev/i2c-0"
    // 模型目录
    #define MODEL_ROOT_DIR "/emmc/_model"
    // 屏幕
    #define SSD1309
    #define OLED_I2C_ADDR (0x3c)
    // 键盘
    #define KB_I2C_ADDR (0x27)
    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)
    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"
    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    #define MATMUL_PTHREAD

// ===============================================================================
// Nano-ESP: ESP32-S3
// ===============================================================================
#elif defined(NANO_ESP32_S3)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    // #define I2C_DEVFILE "/dev/i2c-1"
    // 模型目录
    // #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"
    // 屏幕
    #define SSD1306
    #define OLED_I2C_ADDR (0x3c)
    // 键盘
    #define KB_I2C_ADDR (0x27)
    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)
    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"
    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

// ===============================================================================
// Nano-ESP: ESP32-P4
// ===============================================================================
#elif defined(NANO_ESP32_P4)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    // #define I2C_DEVFILE "/dev/i2c-1"
    // 模型目录
    // #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"
    // 屏幕
    #define SSD1306
    #define OLED_I2C_ADDR (0x3c)
    // 键盘
    #define KB_I2C_ADDR (0x27)
    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)
    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"
    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

// ===============================================================================
// Nano-TTY: 在终端上模拟Nano-Pod的图形界面和交互
// ===============================================================================
#elif defined(NANO_TTY)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    // #define I2C_DEVFILE "/dev/i2c-1"
    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"
    // 屏幕
    #define NCURSES
    // #define OLED_I2C_ADDR (0x3c)
    // 键盘
    // #define KB_I2C_ADDR (0x27)
    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)
    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"
    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

// ===============================================================================
// Nano-CLI
// ===============================================================================
#elif defined(NANO_CLI)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    // #define I2C_DEVFILE "/dev/i2c-1"
    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"
    // 屏幕
    // #define SSD1309
    // #define OLED_I2C_ADDR (0x3c)
    // 键盘
    // #define KB_I2C_ADDR (0x27)
    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)
    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"
    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

// ===============================================================================
// Nano-Sort
// ===============================================================================
#elif defined(NANO_SORT)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    // #define I2C_DEVFILE "/dev/i2c-1"
    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"
    // 屏幕
    // #define SSD1309
    // #define OLED_I2C_ADDR (0x3c)
    // 键盘
    // #define KB_I2C_ADDR (0x27)
    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)
    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"
    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

// ===============================================================================
// Nano-WSS
// ===============================================================================
#elif defined(NANO_WSS)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    // #define I2C_DEVFILE "/dev/i2c-1"
    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"
    // 屏幕
    // #define SSD1309
    // #define OLED_I2C_ADDR (0x3c)
    // 键盘
    // #define KB_I2C_ADDR (0x27)
    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)
    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"
    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

#else

#endif

#ifdef __cplusplus
}
#endif

#endif
