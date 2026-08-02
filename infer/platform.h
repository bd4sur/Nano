#ifndef __NANO_PLATFORM_H__
#define __NANO_PLATFORM_H__

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// ===============================================================================
// 平台选择
// 默认跟随 Arduino IDE 所选板型自动判定（ARDUINO_M5STACK_* 由 Arduino 核心自动定义）；
// 也可手动取消下方注释之一，强制指定目标平台（优先级高于自动判定）。
// 板型差异一律经 NANO_PLATFORM_* 宏控制，不要在业务代码中散落 #ifdef ARDUINO_*。
// ===============================================================================

// #define NANO_PLATFORM_M5CORE2
// #define NANO_PLATFORM_M5CORES3

#if !defined(NANO_PLATFORM_M5CORE2) && !defined(NANO_PLATFORM_M5CORES3)
    #if defined(ARDUINO_M5STACK_CORES3)
        #define NANO_PLATFORM_M5CORES3
    #else
        #define NANO_PLATFORM_M5CORE2   // 默认（含 ARDUINO_M5STACK_CORE2 及非 Arduino 环境）
    #endif
#endif

// ===============================================================================
// 平台差异参数（由 NANO_PLATFORM_* 宏选择）
// 注意：本头文件同时被纯 C 文件包含，此处只允许出现纯数字/字符串常量，
//       不得引用 ESP-IDF 类型（如 I2S_NUM_0 枚举），需要时在 .cpp 中强制转换。
// ===============================================================================

#if defined(NANO_PLATFORM_M5CORES3)

    // SD 卡 SPI 引脚（CoreS3：SCK=36, MISO=35, MOSI=37, CS=4，同 M5Unified 引脚表）
    #define SD_SPI_CS_PIN    (4)
    #define SD_SPI_SCK_PIN   (36)
    #define SD_SPI_MISO_PIN  (35)
    #define SD_SPI_MOSI_PIN  (37)

    // 显示屏 SPI 时钟（Hz）；0 = 不设置，使用 M5GFX 默认值。
    // Core2 上实测 60MHz 稳定（80MHz 闪屏）；CoreS3 未经真机验证，先用默认值。
    #define DISPLAY_SPI_CLOCK_HZ  (0)

    // 麦克风：ES7210 I2S ADC（标准 I2S 模式，非 PDM），与扬声器共用 I2S_NUM_1
    // 引脚（参照 M5Unified）：MCLK=GPIO0, BCLK=GPIO34, WS=GPIO33, DIN=GPIO14
    #define MIC_I2S_PORT     (1)
    #define MIC_PIN_MCLK     (0)
    #define MIC_PIN_BCLK     (34)
    #define MIC_PIN_WS       (33)
    #define MIC_PIN_DATA_IN  (14)

#else // NANO_PLATFORM_M5CORE2

    // SD 卡 SPI 引脚（Core2：SCK=18, MISO=38, MOSI=23, CS=4）
    #define SD_SPI_CS_PIN    (4)
    #define SD_SPI_SCK_PIN   (18)
    #define SD_SPI_MISO_PIN  (38)
    #define SD_SPI_MOSI_PIN  (23)

    // 显示屏 SPI 时钟（Hz）：Core2 实测 60MHz 稳定（原装 40MHz 的 1.5 倍）
    #define DISPLAY_SPI_CLOCK_HZ  (60000000)

    // 麦克风：PDM 硅麦，与扬声器共用 I2S_NUM_0
    // 引脚（参照 M5Core2 FactoryTest）：CLK(WS)=GPIO0, DATA_IN=GPIO34
    #define MIC_I2S_PORT     (0)
    #define MIC_PIN_CLK      (0)
    #define MIC_PIN_DATA_IN  (34)

#endif

// ===============================================================================
// 全局字符串常量
// ===============================================================================

#define NANO_VERSION "2608"

#define LOG_FILE_PATH "chat.jsonl"

// UI字符串缓冲区最大长度限制
#ifndef UI_STR_BUF_MAX_LENGTH
#define UI_STR_BUF_MAX_LENGTH (16384)
#endif


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

// 内存使用情况查询（字节）
// external：大容量主堆（ESP32上为PSRAM堆，供大块内存分配）；internal：内部RAM堆
uint32_t platform_get_free_heap_size(void);             // 主堆当前空闲总量
uint32_t platform_get_largest_free_block(void);         // 主堆最大连续空闲块
uint32_t platform_get_free_heap_size_internal(void);    // 内部RAM堆当前空闲总量
uint32_t platform_get_largest_free_block_internal(void); // 内部RAM堆最大连续空闲块

// ===============================================================================
// 任务抽象（ESP32 上由 FreeRTOS 实现；其他平台可用 pthread 等实现）
// 注意：句柄为不透明 void*，业务代码不得依赖其具体类型；
//       任务入口函数返回前必须调用 platform_task_delete_self() 自删（不返回）。
// ===============================================================================

typedef void* platform_task_handle_t;
typedef void (*platform_task_func_t)(void *arg);

// 创建任务。stack_bytes 为栈字节数；priority 数值越大优先级越高；
// core >= 0 时绑定到指定核，core < 0 时不绑核。返回 0 成功，负数失败。
int32_t platform_task_create(platform_task_func_t func, const char *name,
                             uint32_t stack_bytes, void *arg, int32_t priority,
                             int32_t core, platform_task_handle_t *out_handle);
// 任务内自删除（不返回）
void platform_task_delete_self(void);
// 按句柄强制删除任务（用于清理兜底）
void platform_task_delete(platform_task_handle_t handle);
// 任务内延时（毫秒）
void platform_task_delay_ms(uint32_t ms);

// 全内存屏障（跨核 SPSC 无锁同步用：保证屏障前的写先于屏障后的写对外可见）
#if defined(__GNUC__)
    #define platform_memory_barrier() __sync_synchronize()
#else
    #define platform_memory_barrier()  // 其他平台按需实现
#endif

// 读取二进制文件到内存缓冲区
int32_t platform_read_file_to_buffer(const char *filepath, uint8_t **buffer, size_t *size);

// 将内存缓冲区写入文件（不存在则创建，存在则截断覆盖）。返回0成功，-1失败
int32_t platform_write_buffer_to_file(const char *filepath, const uint8_t *buffer, size_t size);

// 判断路径是否为目录（1-是，0-否或打开失败）
int32_t platform_is_directory(const char *path);

// 随机访问文件读取（单句柄：同一时刻仅支持一个打开的文件，供电子书等分块读取使用）
int32_t  platform_file_open(const char *filepath);    // 0成功，-1失败
uint32_t platform_file_size(void);                    // 当前打开文件的大小（字节）
int32_t  platform_file_seek(uint32_t offset);         // 0成功，-1失败
int32_t  platform_file_read(uint8_t *buffer, size_t size); // 实际读取字节数，-1失败
void     platform_file_close(void);

// 设置时间
void set_sys_time(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second);

// 初始化文件系统
int32_t fs_init();

int32_t list_files(const char *dir, char **filenames);

// 振动(0-255)
void set_vibration(uint32_t level);

// 全局主音量（0~255）：由 ui_init/系统设置写入（同时应用到扬声器硬件）；
// 音频相关 HAL（如 mic_close 恢复扬声器）读取该值恢复主音量，避免硬编码。
void    platform_set_master_volume(uint8_t volume);
uint8_t platform_get_master_volume(void);









// ===============================================================================
// Nano-Pod-Lite: Raspberry Pi 5
// ===============================================================================
#if defined(NANO_POD_LITE_RPI5)

    // UI字符串缓冲区最大长度限制
    #define UI_STR_BUF_MAX_LENGTH (16384)

    // I2C端口设备文件（屏幕、键盘、UPS共用）
    #define I2C_DEVFILE "/dev/i2c-1"

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

    // SPI设备文件
    #define SPI_DEVFILE "/dev/spidev0.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"

    // 屏幕
    #define USE_DEV_LIB
    #define SCREEN_SPI_CS_CHIP  (4)
    #define SCREEN_SPI_CS_LINE  (17)
    #define SCREEN_SPI_RST_CHIP (4)
    #define SCREEN_SPI_RST_LINE (27)
    #define SCREEN_SPI_DC_CHIP  (4)
    #define SCREEN_SPI_DC_LINE  (22)
    #define SCREEN_SPI_BL_CHIP  (4)
    #define SCREEN_SPI_BL_LINE  (25)

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

    // SPI设备文件
    #define SPI_DEVFILE "/dev/spidev1.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/home/radxa/ai/_model/Nano"

    // 屏幕
    #define USE_DEV_LIB
    #define SCREEN_SPI_CS_CHIP  (0)
    #define SCREEN_SPI_CS_LINE  (38)
    #define SCREEN_SPI_RST_CHIP (0)
    #define SCREEN_SPI_RST_LINE (36)
    #define SCREEN_SPI_DC_CHIP  (0)
    #define SCREEN_SPI_DC_LINE  (39)
    #define SCREEN_SPI_BL_CHIP  (0)
    #define SCREEN_SPI_BL_LINE  (40)

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

    // SPI设备文件
    #define SPI_DEVFILE "/dev/spidev1.0"

    // 模型目录
    #define MODEL_ROOT_DIR "/root/ai/_model/Nano"

    // 屏幕
    #define USE_DEV_LIB
    #define SCREEN_WIDTH  (320)
    #define SCREEN_HEIGHT (240)
    #define SCREEN_SPI_CS_CHIP  (7)
    #define SCREEN_SPI_CS_LINE  (5)
    #define SCREEN_SPI_RST_CHIP (8)
    #define SCREEN_SPI_RST_LINE (0)
    #define SCREEN_SPI_DC_CHIP  (7)
    #define SCREEN_SPI_DC_LINE  (4)

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
