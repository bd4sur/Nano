#include "platform.h"
#include "hal_fs.h"
#include "hal_ram.h" // list_files / platform_read_file_to_buffer 内部经 platform_malloc 分配缓冲

#include <Arduino.h>
#include <SPI.h>
#include <SD.h>
#include <string.h>

// SD 卡 SPI 引脚宏（SD_SPI_*_PIN）在 platform.h 中按 NANO_PLATFORM_* 定义

int32_t fs_init() {
    // SD Card Initialization
    SPI.begin(SD_SPI_SCK_PIN, SD_SPI_MISO_PIN, SD_SPI_MOSI_PIN, SD_SPI_CS_PIN);

    if (!SD.begin(SD_SPI_CS_PIN, SPI, 25000000)) {
        printf("Card failed, or not present");
        return -1;
    }

    uint64_t cardSize = SD.cardSize() / (1024 * 1024);
    printf("SD Card Size: %lluMB\n", cardSize);

    return 0;
}


int32_t platform_read_file_to_buffer(const char *filepath, uint8_t **buffer, size_t *size) {
    if (!SD.exists(filepath)) {
        printf("!SD.exists(filepath)");
        return -1;
    }

    File file = SD.open(filepath, FILE_READ);
    if (!file) {
        printf("!file");
        return -1;
    }

    size_t fileSize = file.size();
    *buffer = (uint8_t *)platform_malloc(fileSize);
    if (*buffer == NULL) {
        printf("*buffer == NULL");
        file.close();
        return -1;
    }

    size_t bytesRead = file.read(*buffer, fileSize);
    file.close();

    if (bytesRead != fileSize) {
        printf("bytesRead != fileSize");
        free(*buffer);
        *buffer = NULL;
        return -1;
    }

    *size = fileSize;
    return 0;
}

int32_t platform_write_buffer_to_file(const char *filepath, const uint8_t *buffer, size_t size) {
    File file = SD.open(filepath, FILE_WRITE); // 不存在则创建，存在则截断
    if (!file) {
        return -1;
    }
    size_t written = file.write(buffer, size);
    file.close();
    return (written == size) ? 0 : -1;
}

int32_t platform_is_directory(const char *path) {
    File f = SD.open(path);
    if (!f) {
        return 0;
    }
    int32_t is_dir = f.isDirectory() ? 1 : 0;
    f.close();
    return is_dir;
}

int32_t platform_mkdir(const char *path) {
    if (platform_is_directory(path)) {
        return 0; // 已存在且为目录
    }
    return SD.mkdir(path) ? 0 : -1;
}

// 随机访问文件读取（单句柄）
static File s_platform_file;

int32_t platform_file_open(const char *filepath) {
    if (s_platform_file) {
        s_platform_file.close();
    }
    s_platform_file = SD.open(filepath, FILE_READ);
    return s_platform_file ? 0 : -1;
}

uint32_t platform_file_size(void) {
    return s_platform_file ? s_platform_file.size() : 0;
}

int32_t platform_file_seek(uint32_t offset) {
    return (s_platform_file && s_platform_file.seek(offset)) ? 0 : -1;
}

int32_t platform_file_read(uint8_t *buffer, size_t size) {
    if (!s_platform_file) {
        return -1;
    }
    return (int32_t)s_platform_file.read(buffer, size);
}

void platform_file_close(void) {
    if (s_platform_file) {
        s_platform_file.close();
    }
}

/**
 * 列出目录中的文件（纯 C 接口）
 *
 * @param dir       目录路径，如 "/" 或 "/data"
 * @param filenames 文件名字符串指针数组。
 *                  传 NULL 时仅返回数量，不分配内存；
 *                  非 NULL 时按顺序填充文件名（需预先分配 count 个 char*）
 * @return  >=0: 文件数量
 *           -1: 目录打开失败或路径不是目录
 *           -2: 内存分配失败（仅当 filenames!=NULL 时可能返回）
 */
int32_t list_files(const char *dir, char **filenames)
{
    File root = SD.open(dir);
    if (!root || !root.isDirectory()) {
        return -1;
    }

    int32_t count = 0;
    File entry = root.openNextFile();

    while (entry) {
        if (filenames != NULL) {
            const char *src = entry.name();          // ESP32 返回 const char*
            size_t len = strlen(src);

            filenames[count] = (char *)platform_malloc(len + 1); // 文件名缓冲分配在PSRAM
            if (filenames[count] == NULL) {
                /* 分配失败：回滚已分配的内存，避免泄漏 */
                for (int32_t i = 0; i < count; i++) {
                    free(filenames[i]);
                    filenames[i] = NULL;
                }
                entry.close();
                root.close();
                return -2;
            }
            memcpy(filenames[count], src, len + 1);  // 含 '\0'
        }
        count++;
        entry.close();          // 必须及时关闭，释放文件句柄
        entry = root.openNextFile();
    }

    root.close();
    return count;
}

// 主函数：将 prompt 和 response 转义、转换、写入 log.jsonl
int32_t write_chat_log(char *filepath, uint64_t timestamp, wchar_t* prompt, wchar_t* response) {
    // Stub
    return 0;
}

// 读取文件内容（UTF-8），并转换为 wchar_t* 字符串
wchar_t* read_file_to_wchar(char* filename) {
    // Stub
    return NULL;
}
