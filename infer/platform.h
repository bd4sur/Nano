#ifndef __NANO_PLATFORM_H__
#define __NANO_PLATFORM_H__

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// ===============================================================================
// 全局字符串常量
// ===============================================================================

#define NANO_VERSION "2605"

#define LOG_FILE_PATH "chat.jsonl"

// ===============================================================================
// 平台相关工具函数
// ===============================================================================

void sleep_in_ms(uint32_t ms);
uint64_t get_timestamp_in_ms();
int32_t graceful_shutdown();

// 将对话记录写入日志文件（JSONL格式）
int32_t write_chat_log(char *filepath, uint64_t timestamp, wchar_t* prompt, wchar_t* response);
// 读取文件，并返回新的wchar数组
wchar_t* read_file_to_wchar(char* filename);

// 根据设备类型选择不同的 m/calloc 实现
void *platform_calloc(size_t n, size_t sizeoftype);
void *platform_calloc_internal(size_t n, size_t sizeoftype);
void *platform_malloc(size_t nbytes);
void *platform_malloc_internal(size_t nbytes);
void *platform_realloc(void *ptr, size_t n);
void *platform_realloc_internal(void *ptr, size_t n);

// 读取二进制文件到内存缓冲区
int32_t platform_read_file_to_buffer(const char *filepath, uint8_t **buffer, size_t *size);



// ===============================================================================
// Nano-Pod-Lite: Raspberry Pi 5
// ===============================================================================
#if defined(NANO_POD_LITE_RPI5)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (16384)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    #define I2C_DEVFILE "/dev/i2c-1"

    // GPIO设备文件
    #define GPIO_CHIP_DEVFILE "/dev/gpiochip4"

    // SPI设备文件
    #define SPI_DEVFILE "/dev/spidev0.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"

    // 屏幕
    #define SSD1306
    #define OLED_I2C_ADDR (0x3c)
    #define SCREEN_WIDTH  (128)
    #define SCREEN_HEIGHT (64)

    // 键盘
    #define KB_I2C_ADDR (0x27)

    // 蜂鸣器
    // #define BUZZER_ENABLED
    #define BUZZER_GPIO 6

    // IMU
    // #define IMU_ENABLED
    #define IMU_DEVFILE "/dev/ttyAMA0"

    // UPS
    #define UPS_ENABLED
    #define UPS_I2C_ADDR (0x36)

    // ASR和TTS
    #define ASR_ENABLED
    #define TTS_ENABLED
    #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"

    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

    // BadApple
    #define BADAPPLE_ENABLED


// ===============================================================================
// Nano-Pod: Raspberry Pi 5 + SPI LCD + IMU
// ===============================================================================
#elif defined(NANO_POD_RPI5)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (16384)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    #define I2C_DEVFILE "/dev/i2c-1"

    // GPIO设备文件
    #define GPIO_CHIP_DEVFILE "/dev/gpiochip4"

    // SPI设备文件
    #define SPI_DEVFILE "/dev/spidev0.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"

    // 屏幕
    #define USE_DEV_LIB
    #define SCREEN_SPI_CS  17
    #define SCREEN_SPI_RST 27
    #define SCREEN_SPI_DC  22
    #define SCREEN_SPI_BL  25
    #define OLED_I2C_ADDR (0x3c)
    #define SCREEN_WIDTH  (320)
    #define SCREEN_HEIGHT (240)

    // 键盘
    #define KB_I2C_ADDR (0x27)

    // 蜂鸣器
    // #define BUZZER_ENABLED
    #define BUZZER_GPIO 6

    // IMU
    #define IMU_ENABLED
    #define IMU_DEVFILE "/dev/ttyAMA0"

    // UPS
    #define UPS_ENABLED
    #define UPS_I2C_ADDR (0x36)

    // ASR和TTS
    #define ASR_ENABLED
    #define TTS_ENABLED
    #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"

    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

    // BadApple
    #define BADAPPLE_ENABLED


// ===============================================================================
// Nano-Pod-Lite: Rock 5B+
// ===============================================================================
#elif defined(NANO_POD_LITE_ROCK5BP)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (16384)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    #define I2C_DEVFILE "/dev/i2c-3"

    // GPIO设备文件
    #define GPIO_CHIP_DEVFILE "/dev/gpiochip0"

    // SPI设备文件
    #define SPI_DEVFILE "/dev/spidev0.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"

    // 屏幕
    #define SSD1309
    #define OLED_I2C_ADDR (0x3c)
    #define SCREEN_WIDTH  (128)
    #define SCREEN_HEIGHT (64)

    // 键盘
    #define KB_I2C_ADDR (0x27)

    // 蜂鸣器
    // #define BUZZER_ENABLED
    #define BUZZER_GPIO 6

    // IMU
    // #define IMU_ENABLED
    #define IMU_DEVFILE "/dev/ttyAMA0"

    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)

    // ASR和TTS
    #define ASR_ENABLED
    #define TTS_ENABLED
    #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"

    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

    // BadApple
    #define BADAPPLE_ENABLED


// ===============================================================================
// Nano-Pod-Lite: Cubie-A7Z
// ===============================================================================
#elif defined(NANO_POD_LITE_CUBIE_A7Z)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (16384)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    #define I2C_DEVFILE "/dev/i2c-7"

    // GPIO设备文件
    #define GPIO_CHIP_DEVFILE "/dev/gpiochip0"

    // SPI设备文件
    #define SPI_DEVFILE "/dev/spidev1.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/home/radxa/ai/_model/Nano"

    // 屏幕
    #define SSD1306
    #define OLED_I2C_ADDR (0x3c)
    #define SCREEN_WIDTH  (128)
    #define SCREEN_HEIGHT (64)

    // 键盘
    #define KB_I2C_ADDR (0x27)

    // 蜂鸣器
    // #define BUZZER_ENABLED
    #define BUZZER_GPIO 6

    // IMU
    // #define IMU_ENABLED
    #define IMU_DEVFILE "/dev/ttyAMA0"

    // UPS
    // #define UPS_ENABLED
    #define UPS_I2C_ADDR (0x36)

    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"

    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

    // BadApple
    #define BADAPPLE_ENABLED


// ===============================================================================
// Nano-Pod: Cubie-A7Z + SPI LCD + IMU
// ===============================================================================
#elif defined(NANO_POD_CUBIE_A7Z)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (16384)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    #define I2C_DEVFILE "/dev/i2c-7"

    // GPIO设备文件
    #define GPIO_CHIP_DEVFILE "/dev/gpiochip0"

    // SPI设备文件
    #define SPI_DEVFILE "/dev/spidev1.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/home/radxa/ai/_model/Nano"

    // 屏幕
    #define USE_DEV_LIB
    #define SCREEN_SPI_CS  38
    #define SCREEN_SPI_RST 36
    #define SCREEN_SPI_DC  39
    #define SCREEN_SPI_BL  40
    #define OLED_I2C_ADDR (0x3c)
    #define SCREEN_WIDTH  (320)
    #define SCREEN_HEIGHT (240)

    // 键盘
    #define KB_I2C_ADDR (0x27)

    // 蜂鸣器
    // #define BUZZER_ENABLED
    #define BUZZER_GPIO 6

    // IMU
    #define IMU_ENABLED
    #define IMU_DEVFILE "/dev/ttyAS4"

    // UPS
    // #define UPS_ENABLED
    #define UPS_I2C_ADDR (0x36)

    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"

    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

    // BadApple
    #define BADAPPLE_ENABLED


// ===============================================================================
// Nano-Pod: STM32MP135 + SPI LCD + IMU
// ===============================================================================
#elif defined(NANO_POD_MP135)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (16384)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    #define I2C_DEVFILE "/dev/i2c-1"

    // GPIO设备文件
    #define GPIO_CHIP_DEVFILE "/dev/gpiochip0"

    // SPI设备文件
    #define SPI_DEVFILE "/dev/spidev1.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/root/ai/_model/Nano"

    // 屏幕
    #define USE_DEV_LIB
    #define SCREEN_WIDTH  (320)
    #define SCREEN_HEIGHT (240)

    // 键盘
    #define KB_I2C_ADDR (0x27)

    // 蜂鸣器
    // #define BUZZER_ENABLED
    // #define BUZZER_GPIO 6

    // IMU
    // #define IMU_ENABLED
    // #define IMU_DEVFILE "/dev/ttyAS4"

    // UPS
    #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)

    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"

    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

    // BadApple
    #define BADAPPLE_ENABLED


// ===============================================================================
// Nano-Pod-Lite: Make Router Great Again 京东云RE-CS-02、红米AX5等
// ===============================================================================
#elif defined(NANO_POD_LITE_MARGA)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (16384)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    #define I2C_DEVFILE "/dev/i2c-0"

    // GPIO设备文件
    #define GPIO_CHIP_DEVFILE "/dev/gpiochip4"

    // SPI设备文件
    #define SPI_DEVFILE "/dev/spidev0.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/emmc/_model"

    // 屏幕
    #define SSD1309
    #define OLED_I2C_ADDR (0x3c)
    #define SCREEN_WIDTH  (128)
    #define SCREEN_HEIGHT (64)

    // 键盘
    #define KB_I2C_ADDR (0x27)

    // 蜂鸣器
    // #define BUZZER_ENABLED
    #define BUZZER_GPIO 6

    // IMU
    // #define IMU_ENABLED
    #define IMU_DEVFILE "/dev/ttyAMA0"

    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)

    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"

    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    #define MATMUL_PTHREAD

    // BadApple
    #define BADAPPLE_ENABLED


// ===============================================================================
// Nano-ESP: ESP32-S3
// ===============================================================================
#elif defined(NANO_ESP32_S3)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (1024)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    // #define I2C_DEVFILE "/dev/i2c-1"

    // GPIO设备文件
    // #define GPIO_CHIP_DEVFILE "/dev/gpiochip0"

    // SPI设备文件
    // #define SPI_DEVFILE "/dev/spidev0.0"

    // 模型目录
    // #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"

    // 屏幕
    #define SSD1306
    #define OLED_I2C_ADDR (0x3c)
    #define SCREEN_WIDTH  (128)
    #define SCREEN_HEIGHT (64)

    // 键盘
    #define KB_I2C_ADDR (0x27)

    // 蜂鸣器
    // #define BUZZER_ENABLED
    #define BUZZER_GPIO 6

    // IMU
    // #define IMU_ENABLED
    // #define IMU_DEVFILE "/dev/ttyAMA0"

    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)

    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"

    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

    // BadApple
    #define BADAPPLE_ENABLED


// ===============================================================================
// Nano-ESP: ESP32-P4
// ===============================================================================
#elif defined(NANO_ESP32_P4)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (1024)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    // #define I2C_DEVFILE "/dev/i2c-1"

    // GPIO设备文件
    // #define GPIO_CHIP_DEVFILE "/dev/gpiochip0"

    // SPI设备文件
    // #define SPI_DEVFILE "/dev/spidev0.0"

    // 模型目录
    // #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"

    // 屏幕
    #define SSD1306
    #define OLED_I2C_ADDR (0x3c)
    #define SCREEN_WIDTH  (128)
    #define SCREEN_HEIGHT (64)

    // 键盘
    #define KB_I2C_ADDR (0x27)

    // 蜂鸣器
    // #define BUZZER_ENABLED
    #define BUZZER_GPIO 6

    // IMU
    // #define IMU_ENABLED
    // #define IMU_DEVFILE "/dev/ttyAMA0"

    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)

    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"

    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

    // BadApple
    #define BADAPPLE_ENABLED


// ===============================================================================
// Nano-TTY: 在终端上模拟Nano-Pod的图形界面和交互
// ===============================================================================
#elif defined(NANO_TTY)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (16384)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    // #define I2C_DEVFILE "/dev/i2c-1"

    // GPIO设备文件
    // #define GPIO_CHIP_DEVFILE "/dev/gpiochip0"

    // SPI设备文件
    // #define SPI_DEVFILE "/dev/spidev0.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"

    // 屏幕
    #define NCURSES
    // #define OLED_I2C_ADDR (0x3c)
    #define SCREEN_WIDTH  (320)
    #define SCREEN_HEIGHT (240)

    // 键盘
    // #define KB_I2C_ADDR (0x27)

    // 蜂鸣器
    // #define BUZZER_ENABLED
    // #define BUZZER_GPIO 6

    // IMU
    // #define IMU_ENABLED
    // #define IMU_DEVFILE "/dev/ttyAMA0"

    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)

    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"

    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

    // BadApple
    #define BADAPPLE_ENABLED


// ===============================================================================
// Nano-CLI
// ===============================================================================
#elif defined(NANO_CLI)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (16384)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    // #define I2C_DEVFILE "/dev/i2c-1"

    // GPIO设备文件
    // #define GPIO_CHIP_DEVFILE "/dev/gpiochip0"

    // SPI设备文件
    // #define SPI_DEVFILE "/dev/spidev0.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"

    // 屏幕
    // #define SSD1309
    // #define OLED_I2C_ADDR (0x3c)
    // #define SCREEN_WIDTH  (128)
    // #define SCREEN_HEIGHT (64)

    // 键盘
    // #define KB_I2C_ADDR (0x27)

    // 蜂鸣器
    // #define BUZZER_ENABLED
    // #define BUZZER_GPIO 6

    // IMU
    // #define IMU_ENABLED
    // #define IMU_DEVFILE "/dev/ttyAMA0"

    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)

    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"

    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

    // BadApple
    // #define BADAPPLE_ENABLED


// ===============================================================================
// Nano-Sort
// ===============================================================================
#elif defined(NANO_SORT)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (16384)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    // #define I2C_DEVFILE "/dev/i2c-1"

    // GPIO设备文件
    // #define GPIO_CHIP_DEVFILE "/dev/gpiochip0"

    // SPI设备文件
    // #define SPI_DEVFILE "/dev/spidev0.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"

    // 屏幕
    // #define SSD1309
    // #define OLED_I2C_ADDR (0x3c)
    // #define SCREEN_WIDTH  (128)
    // #define SCREEN_HEIGHT (64)

    // 键盘
    // #define KB_I2C_ADDR (0x27)

    // 蜂鸣器
    // #define BUZZER_ENABLED
    // #define BUZZER_GPIO 6

    // IMU
    // #define IMU_ENABLED
    // #define IMU_DEVFILE "/dev/ttyAMA0"

    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)

    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"

    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

    // BadApple
    // #define BADAPPLE_ENABLED


// ===============================================================================
// Nano-WSS
// ===============================================================================
#elif defined(NANO_WSS)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (16384)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    // #define I2C_DEVFILE "/dev/i2c-1"

    // GPIO设备文件
    // #define GPIO_CHIP_DEVFILE "/dev/gpiochip0"

    // SPI设备文件
    // #define SPI_DEVFILE "/dev/spidev0.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"

    // 屏幕
    // #define SSD1309
    // #define OLED_I2C_ADDR (0x3c)
    // #define SCREEN_WIDTH  (128)
    // #define SCREEN_HEIGHT (64)

    // 键盘
    // #define KB_I2C_ADDR (0x27)

    // 蜂鸣器
    // #define BUZZER_ENABLED
    // #define BUZZER_GPIO 6

    // IMU
    // #define IMU_ENABLED
    // #define IMU_DEVFILE "/dev/ttyAMA0"

    // UPS
    // #define UPS_ENABLED
    // #define UPS_I2C_ADDR (0x36)

    // ASR和TTS
    // #define ASR_ENABLED
    // #define TTS_ENABLED
    // #define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"

    // 是否使用pthread实现的matmul？（用于OpenWrt等对OpenMP不友好的场景）
    // #define MATMUL_PTHREAD

    // BadApple
    // #define BADAPPLE_ENABLED

#else

#endif

#ifdef __cplusplus
}
#endif

#endif
