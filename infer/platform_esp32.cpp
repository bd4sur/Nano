#include "platform.h"

#include <Arduino.h>
#include <esp32-hal-psram.h>
#include <Wire.h>

extern "C" {

void sleep_in_ms(uint32_t ms) {
    delay(ms);
}

uint64_t get_timestamp_in_ms() {
    return (uint64_t)millis();
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


void print_str(char* msg) {
    Serial.print(msg);
}

void print_num(int i) {
    Serial.print(i);
}

void print_float(float i) {
    Serial.print(i);
}

void *psram_calloc(size_t n, size_t sizeoftype) {
    return ps_calloc(n, sizeoftype);
}


}
