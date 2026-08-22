#ifndef __NANO_HAL_FS_H__
#define __NANO_HAL_FS_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===============================================================================
// hal_fs：文件系统抽象层（硬件无关接口；ESP32 实现见 hal_fs_m5esp.cpp）
//   SD/文件系统根目录前缀见 platform.h 的 PLATFORM_ROOT_DIR；
//   本层函数仅依赖 hal_ram 的内存分配器，各功能模块可直接包含本头文件使用。
// ===============================================================================

// 初始化文件系统（SD 卡）。返回 0 成功，-1 失败
int32_t fs_init();

// 读取二进制文件到内存缓冲区（缓冲经 platform_malloc 分配于 PSRAM，调用方负责 free）
int32_t platform_read_file_to_buffer(const char *filepath, uint8_t **buffer, size_t *size);

// 将内存缓冲区写入文件（不存在则创建，存在则截断覆盖）。返回0成功，-1失败
int32_t platform_write_buffer_to_file(const char *filepath, const uint8_t *buffer, size_t size);

// 判断路径是否为目录（1-是，0-否或打开失败）
int32_t platform_is_directory(const char *path);

// 创建目录（已存在且为目录则视为成功）。返回0成功，-1失败
int32_t platform_mkdir(const char *path);

// 随机访问文件读取（单句柄：同一时刻仅支持一个打开的文件，供电子书/词典等分块读取使用）
int32_t  platform_file_open(const char *filepath);    // 0成功，-1失败
uint32_t platform_file_size(void);                    // 当前打开文件的大小（字节）
int32_t  platform_file_seek(uint32_t offset);         // 0成功，-1失败
int32_t  platform_file_read(uint8_t *buffer, size_t size); // 实际读取字节数，-1失败
void     platform_file_close(void);

/**
 * 列出目录中的文件（纯 C 接口，文件名缓冲经 platform_malloc 分配于 PSRAM）
 *
 * @param dir       目录路径，如 "/" 或 "/data"
 * @param filenames 文件名字符串指针数组。
 *                  传 NULL 时仅返回数量，不分配内存；
 *                  非 NULL 时按顺序填充文件名（需预先分配 count 个 char*）
 * @return  >=0: 文件数量
 *           -1: 目录打开失败或路径不是目录
 *           -2: 内存分配失败（仅当 filenames!=NULL 时可能返回）
 */
int32_t list_files(const char *dir, char **filenames);

// 将对话记录写入日志文件（JSONL格式）
int32_t write_chat_log(char *filepath, uint64_t timestamp, wchar_t* prompt, wchar_t* response);
// 读取文件，并返回新的wchar数组
wchar_t* read_file_to_wchar(char* filename);

#ifdef __cplusplus
}
#endif

#endif
