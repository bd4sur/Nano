//
// nano_min_esp32.cpp - nano_min 引擎文件原语的 ESP32 实现（Arduino SD 库，C++ 垫片）
//
//   为什么需要本文件：nano_min.c 是纯 C，而部分 Arduino-ESP32 内核的 SD 库并未把
//   "/sdcard" 挂载点注册进 VFS（fopen("/sdcard/...") 报 ENOENT），设备上唯一可靠的
//   文件途径是 SD 库 API 本身（与 platform_esp32.cpp 一致）。SD.h 为 C++，故有此垫片。
//
//   路径语义：直接使用业务路径（SD 卡根目录相对，如 "/llm/nano_min_work.tmp"）。
//   调用前应用须已完成 fs_init()（即 SD.begin）。
//

#include <SD.h>
#include <stdio.h>

extern "C" {

// rw=0：只读打开；rw=1：读写打开（不截断——KV/logits 区域按位置整体覆写，
// 读取范围不会超过已写入范围，因此无需截断）
void *nm_sd_open(const char *path, int32_t rw) {
    File *fp = new File();
    if (rw) {
        // 注意：Arduino FS 的 "r+" 不会创建文件（其 create 参数仅创建目录），须先以 "w" 创建
        if (!SD.exists(path)) {
            File fc = SD.open(path, FILE_WRITE);
            if (fc) fc.close();
            if (!SD.exists(path)) {
                fprintf(stderr, "nano_min: create %s failed (dir /llm exists=%d)\n", path, SD.exists("/llm") ? 1 : 0);
            }
        }
        *fp = SD.open(path, "r+");
    }
    else {
        *fp = SD.open(path, FILE_READ);
    }
    if (!(*fp)) {
        fprintf(stderr, "nano_min: SD.open %s (rw=%d) failed\n", path, (int)rw);
        delete fp;
        return NULL;
    }
    return (void *)fp;
}

int32_t nm_sd_pread(void *f, void *buf, uint32_t size, uint64_t offset) {
    File *fp = (File *)f;
    if (!fp->seek((uint32_t)offset)) return -1;
    uint8_t *p = (uint8_t *)buf;
    uint32_t done = 0;
    while (done < size) {
        int r = fp->read(p + done, size - done);
        if (r <= 0) return -1;
        done += (uint32_t)r;
    }
    return 0;
}

int32_t nm_sd_pwrite(void *f, const void *buf, uint32_t size, uint64_t offset) {
    File *fp = (File *)f;
    if (!fp->seek((uint32_t)offset)) return -1;
    size_t w = fp->write((const uint8_t *)buf, size);
    fp->flush(); // 保证后续读立即可见
    return (w == size) ? 0 : -1;
}

void nm_sd_close(void *f) {
    if (!f) return;
    File *fp = (File *)f;
    fp->close();
    delete fp;
}

int32_t nm_sd_remove(const char *path) {
    return SD.remove(path) ? 0 : -1;
}

int32_t nm_sd_exists(const char *path) {
    return SD.exists(path) ? 1 : 0;
}

}
