#include "platform.h"

#include <sys/time.h>
#include <Arduino.h>
#include <esp32-hal-psram.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <M5Unified.h>
#include <SPI.h>
#include <SD.h>

extern "C" {

void sleep_in_ms(uint32_t ms) {
    delay(ms);
}

uint64_t get_timestamp_in_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + (uint64_t)tv.tv_usec / 1000;
}

// 优雅关机：通过 PMIC（Core2: AXP192 / CoreS3: AXP2101，M5.Power 已封装板型差异）切断整机全部电源。
// 成功时本函数不返回（设备已断电）；若仍然返回，说明断电未生效（如 USB 供电时），
// 返回 -1 由调用方提示关机失败。
int32_t graceful_shutdown() {
    sleep_in_ms(500); // 让“正在安全关机”提示在屏幕上停留片刻
    M5.Power.powerOff(); // 切断所有电源输出
    sleep_in_ms(1000); // 等待断电生效；仍在运行则判定失败
    return -1;
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

void set_sys_time(
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second
) {
    M5.Rtc.setDateTime( { { year, month, day }, { hour, minute, second } } );
}

void *platform_calloc(size_t n, size_t sizeoftype) {
    return heap_caps_calloc((n), (sizeoftype), MALLOC_CAP_SPIRAM);
}

void *platform_calloc_internal(size_t n, size_t sizeoftype) {
    return heap_caps_calloc((n), (sizeoftype), MALLOC_CAP_DEFAULT);
}

void *platform_malloc(size_t nbytes) {
    return heap_caps_malloc((nbytes), MALLOC_CAP_SPIRAM);
}

void *platform_malloc_internal(size_t nbytes) {
    return heap_caps_malloc((nbytes), MALLOC_CAP_DEFAULT);
}

void *platform_realloc(void *ptr, size_t n) {
    return heap_caps_realloc((ptr), (n), MALLOC_CAP_SPIRAM);
}

void *platform_realloc_internal(void *ptr, size_t n) {
    return heap_caps_realloc((ptr), (n), MALLOC_CAP_DEFAULT);
}

uint32_t platform_get_free_heap_size() {
    return heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}

uint32_t platform_get_largest_free_block() {
    return heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
}

uint32_t platform_get_free_heap_size_internal() {
    return heap_caps_get_free_size(MALLOC_CAP_DEFAULT);
}

uint32_t platform_get_largest_free_block_internal() {
    return heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT);
}

// ---------------- 任务抽象（FreeRTOS 实现） ----------------

int32_t platform_task_create(platform_task_func_t func, const char *name,
                             uint32_t stack_bytes, void *arg, int32_t priority,
                             int32_t core, platform_task_handle_t *out_handle) {
    TaskHandle_t handle = NULL;
    BaseType_t ok;
    if (core >= 0) {
        ok = xTaskCreatePinnedToCore(func, name, stack_bytes, arg,
                                     (UBaseType_t)priority, &handle, (BaseType_t)core);
    }
    else {
        ok = xTaskCreate(func, name, stack_bytes, arg,
                         (UBaseType_t)priority, &handle);
    }
    if (ok != pdPASS) return -1;
    if (out_handle) *out_handle = (platform_task_handle_t)handle;
    return 0;
}

void platform_task_delete_self(void) {
    vTaskDelete(NULL);
}

void platform_task_delete(platform_task_handle_t handle) {
    if (handle) vTaskDelete((TaskHandle_t)handle);
}

void platform_task_delay_ms(uint32_t ms) {
    vTaskDelay(pdMS_TO_TICKS(ms));
}


}


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




// 振动(0-255)
void set_vibration(uint32_t level) {
    M5.Power.setVibration(level);
}

static uint8_t s_master_volume = 16; // 与 ui_init 的 volume 初值一致

void platform_set_master_volume(uint8_t volume) {
    s_master_volume = volume;
    if (M5.Speaker.isEnabled()) {
        M5.Speaker.setVolume(volume);
    }
}

uint8_t platform_get_master_volume(void) {
    return s_master_volume;
}



























bool playWAVFromSD(const char* filename, uint32_t repeat = 1, int channel = -1, bool stop_current = true);
bool playWAVMemory(File& wavFile, size_t fileSize, uint32_t repeat, int channel, bool stop_current);
bool playWAVSegmented(const char* filename, uint32_t repeat, int channel, bool stop_current);

void play_badapple() {
    fs_init();
    printf("Starting audio playback...");
    M5.Speaker.setVolume(32);
    playWAVFromSD("/badapple.wav", 1, -1, true);
}

bool playWAVFromSD(const char* filename, uint32_t repeat, int channel, bool stop_current) {
    if (!SD.exists(filename)) {
        printf("File does not exist!");
        return false;
    }

    File wavFile = SD.open(filename, FILE_READ);
    if (!wavFile) {
        printf("Failed to open file!");
        return false;
    }

    size_t fileSize = wavFile.size();
    printf("File size: %d Byte (%.2f KB)\n", fileSize, fileSize/1024.0);

    size_t freeHeap = ESP.getFreeHeap();
    printf("Free heap: %d bytes\n", freeHeap);

    if (fileSize < freeHeap / 2) {
        return playWAVMemory(wavFile, fileSize, repeat, channel, stop_current);
    }
    // 文件太大，使用分段播放
    wavFile.close();
    return playWAVSegmented(filename, repeat, channel, stop_current);
}

bool playWAVMemory(File& wavFile, size_t fileSize, uint32_t repeat, int channel, bool stop_current) {
    // 音频数据缓冲分配在PSRAM（M5Unified混音任务以CPU读取源数据，仅内部DMA缓冲需要内部RAM）
    uint8_t* wavData = (uint8_t*)platform_malloc(fileSize);
    if (!wavData) {
        printf("Memory allocation failed!");
        wavFile.close();
        return false;
    }

    printf("Loading file to memory...");
    size_t bytesRead = wavFile.read(wavData, fileSize);
    wavFile.close();

    if (bytesRead != fileSize) {
        printf("Read error: %d/%d bytes\n", bytesRead, fileSize);
        free(wavData);
        return false;
    }

    printf("Starting playback...");
    bool result = M5.Speaker.playWav(wavData, fileSize, repeat, channel, stop_current);

    if (result) {
        while (M5.Speaker.isPlaying()) {
            delay(100);
        }
        printf("Playback completed!");
    }

    free(wavData);
    return result;
}

// 分段播放（大文件）- 优化内存使用
bool playWAVSegmented(const char* filename, uint32_t repeat, int channel, bool stop_current) {
    File wavFile = SD.open(filename, FILE_READ);
    if (!wavFile) return false;

    uint8_t header[44];
    if (wavFile.read(header, 44) != 44) {
        wavFile.close();
        return false;
    }

    if (strncmp((char*)header, "RIFF", 4) != 0 ||
        strncmp((char*)header + 8, "WAVE", 4) != 0) {
        printf("Invalid WAV format");
        wavFile.close();
        return false;
    }

    uint32_t totalFileSize = *(uint32_t*)(header + 4) + 8;
    uint32_t sampleRate = *(uint32_t*)(header + 24);
    uint16_t channels = *(uint16_t*)(header + 22);
    uint16_t bitsPerSample = *(uint16_t*)(header + 34);

    printf("WAV: %dHz, %dch, %dbit\n", sampleRate, channels, bitsPerSample);

    size_t dataSize = totalFileSize - 44;

    size_t bytesPerSample = (bitsPerSample / 8) * channels;
    size_t chunkSizeInSamples = 16384 / bytesPerSample;
    size_t chunkSize = chunkSizeInSamples * bytesPerSample;

    printf("Chunk size: %d bytes (%d samples)\n", chunkSize, chunkSizeInSamples);

    uint8_t* chunkBuffer = nullptr;
    size_t actualChunkSize = chunkSize;

    while (actualChunkSize >= 4096 && !chunkBuffer) {  // 最小4KB
        chunkBuffer = (uint8_t*)malloc(actualChunkSize + 44);
        if (!chunkBuffer) {
            actualChunkSize /= 2;
            chunkSizeInSamples = actualChunkSize / bytesPerSample;
            actualChunkSize = chunkSizeInSamples * bytesPerSample;
            printf("Retrying with smaller chunk: %d bytes\n", actualChunkSize);
        }
    }

    if (!chunkBuffer) {
        printf("Buffer allocation failed even with small chunks!");
        wavFile.close();
        return false;
    }

    printf("Using chunk size: %d bytes\n", actualChunkSize);
    printf("Starting segmented playback...");

    for (uint32_t rep = 0; rep < repeat; rep++) {
        size_t totalRead = 0;
        int segmentNum = 0;

        wavFile.seek(44);

        while (totalRead < dataSize) {
            size_t bytesToRead = min(actualChunkSize, dataSize - totalRead);

            memcpy(chunkBuffer, header, 44);
            uint32_t chunkFileSize = bytesToRead + 36;
            memcpy(chunkBuffer + 4, &chunkFileSize, 4);
            memcpy(chunkBuffer + 40, &bytesToRead, 4);

            size_t bytesRead = wavFile.read(chunkBuffer + 44, bytesToRead);
            if (bytesRead == 0) break;

            totalRead += bytesRead;
            segmentNum++;

            if (segmentNum % 5 == 1) {
                printf("Segment %d (%.1f%%)\n",
                          segmentNum, (float)totalRead / dataSize * 100.0);
            }

            bool playResult = M5.Speaker.playWav(chunkBuffer, bytesRead + 44, 1, channel, stop_current);

            if (!playResult) {
                printf("Segment %d failed\n", segmentNum);
                break;
            }

            while (M5.Speaker.isPlaying()) {
                delay(1);
            }

            delay(1);
        }

        if (rep < repeat - 1) {
            delay(1000);
        }
    }

    free(chunkBuffer);
    wavFile.close();
    printf("Segmented playback completed!");
    return true;
}

