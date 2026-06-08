#include "platform.h"

#include <sys/time.h>
#include <Arduino.h>
#include <esp32-hal-psram.h>

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

// 优雅关机
int32_t graceful_shutdown() {
    return 0;
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


}


#define SD_SPI_CS_PIN   4
#define SD_SPI_SCK_PIN  18
#define SD_SPI_MISO_PIN 38
#define SD_SPI_MOSI_PIN 23

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
    uint8_t* wavData = (uint8_t*)malloc(fileSize);
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

