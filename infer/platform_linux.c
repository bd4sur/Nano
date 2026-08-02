// Linux 平台抽象层实现（PC / 树莓派等普通 Linux 系统）。
//
// 依赖：
//   - pthread（任务抽象基于 pthread 实现，编译/链接需加 -pthread）
//   - 其余均为 glibc / POSIX 标准接口，无额外第三方依赖。
//
// 路径语义说明：与 ESP32（SD 卡根目录 "/"）不同，本实现将路径按原样
// 传递给宿主文件系统，即 "/" 指真实的文件系统根目录。电子书/音乐盒等
// 以 "/" 为根枚举文件的应用，将枚举真实根目录下的文件。

#define _GNU_SOURCE // pthread_setaffinity_np / CPU_ZERO / CPU_SET / timegm

#include "platform.h"

#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <locale.h>
#include <dirent.h>
#include <limits.h> // PTHREAD_STACK_MIN
#include <pthread.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <errno.h>

#ifdef __linux__
#include <sched.h> // CPU_ZERO / CPU_SET / pthread_setaffinity_np
#endif

void sleep_in_ms(uint32_t ms) {
    usleep(ms * 1000);
}

uint64_t get_timestamp_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return (uint64_t)time.tv_sec * 1000ULL + (uint64_t)time.tv_nsec / 1000000ULL;
}

// 优雅关机
int32_t graceful_shutdown() {
    // 同步所有文件系统数据
    sync();
    // 等待同步完成
    sleep(2);
    // 执行关机
    if (system("poweroff") == -1) {
        perror("关机失败");
        return -1;
    }
    return 0;
}


// 辅助函数：在 wchar_t 层面对字符串进行 JSON 转义（\n, ", \）
wchar_t* escape_wchar_string(const wchar_t* wstr) {
    if (!wstr) return NULL;

    size_t len = wcslen(wstr);
    // 最坏情况：每个字符扩展为 2 个宽字符（如 L'\\' → L'\\'L'\\'）
    size_t max_escaped_len = len * 2 + 1;
    wchar_t* escaped = (wchar_t*)calloc(max_escaped_len, sizeof(wchar_t));
    if (!escaped) return NULL;

    wchar_t* out = escaped;
    const wchar_t* p = wstr;

    while (*p) {
        if (*p == L'\n') {
            *out++ = L'\\';
            *out++ = L'n';
        } else if (*p == L'"') {
            *out++ = L'\\';
            *out++ = L'"';
        } else if (*p == L'\\') {
            *out++ = L'\\';
            *out++ = L'\\';
        } else {
            *out++ = *p;
        }
        p++;
    }
    *out = L'\0';
    return escaped;
}

// 将转义后的 wchar_t* 用 wcstombs 转为 char*（依赖当前 locale）
char* wchar_to_utf8(const wchar_t* wstr) {
    if (!wstr) return NULL;

    size_t len = wcslen(wstr);
    size_t buf_size = len * MB_CUR_MAX + 1;
    char* mbstr = (char*)calloc(buf_size, sizeof(char));
    if (!mbstr) return NULL;

    size_t result = _wcstombs(mbstr, wstr, buf_size);
    if (result == (size_t)-1) {
        free(mbstr);
        return NULL;
    }

    return mbstr;
}

// 将 prompt 和 response 转义、转换、写入 log.jsonl
int32_t write_chat_log(char *filepath, uint64_t timestamp, wchar_t* prompt, wchar_t* response) {

    // 第一步：在 wchar_t 层面转义
    wchar_t* escaped_prompt_w = escape_wchar_string(prompt);
    wchar_t* escaped_response_w = escape_wchar_string(response);

    if (!escaped_prompt_w || !escaped_response_w) {
        free(escaped_prompt_w);
        free(escaped_response_w);
        return -1;
    }

    // 第二步：用 wcstombs 转为 char*（UTF-8）
    char* escaped_prompt = wchar_to_utf8(escaped_prompt_w);
    char* escaped_response = wchar_to_utf8(escaped_response_w);

    free(escaped_prompt_w);
    free(escaped_response_w);

    if (!escaped_prompt || !escaped_response) {
        free(escaped_prompt);
        free(escaped_response);
        return -1;
    }

    // 第三步：写入 JSONL 文件
    FILE* fp = fopen(filepath, "a");
    if (!fp) {
        free(escaped_prompt);
        free(escaped_response);
        return -1;
    }

    fprintf(fp, "{\"timestamp\": %ld, \"prompt\": \"%s\", \"response\": \"%s\"}\n",
            timestamp, escaped_prompt, escaped_response);

    fclose(fp);
    free(escaped_prompt);
    free(escaped_response);
    return 0;
}

/**
 * 读取文件内容（UTF-8），并转换为 wchar_t* 字符串
 *
 * @param filename 文件名
 * @return 成功时返回动态分配的 wchar_t*（以 L'\0' 结尾），失败返回 NULL。
 *         调用者需用 free() 释放返回值。
 */
wchar_t* read_file_to_wchar(char* filename) {
    if (!filename) return NULL;

    // 2. 打开文件（当前工作目录）
    FILE* fp = fopen(filename, "rb"); // 用二进制模式避免换行转换
    if (!fp) {
        return NULL;
    }

    // 3. 获取文件大小（可选，用于高效分配）
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    size_t size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }

    // 4. 读取全部内容到 char 缓冲区（+1 保证可加 '\0'）
    char* buffer = (char*)calloc(size + 1, sizeof(char));
    if (!buffer) {
        fclose(fp);
        return NULL;
    }

    size_t bytes_read = fread(buffer, 1, size, fp);
    fclose(fp);

    if ((size_t)bytes_read != size) {
        free(buffer);
        return NULL;
    }
    buffer[size] = '\0'; // 确保以 null 结尾（UTF-8 是 null-safe 的）

    // 5. 计算所需 wchar_t 数量
    size_t wlen = size;

    // 6. 分配 wchar_t 缓冲区
    wchar_t* wstr = (wchar_t*)calloc((wlen + 1), sizeof(wchar_t));
    if (!wstr) {
        free(buffer);
        return NULL;
    }

    // 7. 执行实际转换
    (void)_mbstowcs(wstr, buffer, wlen + 1);
    free(buffer);

    return wstr; // 调用者负责 free()
}

// 设置系统时间（入参为 UTC；对齐 ESP32 M5.Rtc.setDateTime 语义）。
// 需要 root 或 CAP_SYS_TIME 权限，非特权用户调用时静默失败。
void set_sys_time(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second) {
    struct tm t;
    memset(&t, 0, sizeof(t));
    t.tm_year = year - 1900;
    t.tm_mon  = month - 1;
    t.tm_mday = day;
    t.tm_hour = hour;
    t.tm_min  = minute;
    t.tm_sec  = second;

    time_t sec = timegm(&t); // 按 UTC 解释
    if (sec == (time_t)-1) return;

    struct timespec ts;
    ts.tv_sec  = sec;
    ts.tv_nsec = 0;
    (void)clock_settime(CLOCK_REALTIME, &ts); // EPERM 时忽略
}

// Linux 使用宿主原生文件系统，无需初始化
int32_t fs_init() {
    return 0;
}

// 枚举目录下的文件名（不含路径，不含 "." / ".."）。
//   dir       ：目录路径
//   filenames ：为 NULL 时仅返回条目数量；
//               非 NULL 时逐个 platform_malloc 填充文件名（调用方负责 free）。
// 返回值：>=0 条目数量；-1 目录打开失败；-2 内存分配失败（已填充项回滚释放）。
int32_t list_files(const char *dir, char **filenames) {
    if (!dir) return -1;

    DIR *dp = opendir(dir);
    if (!dp) return -1;

    int32_t count = 0;
    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (filenames) {
            size_t len = strlen(entry->d_name);
            char *name = (char *)platform_malloc(len + 1);
            if (!name) { // 分配失败：回滚已填充项
                for (int32_t i = 0; i < count; i++) {
                    free(filenames[i]);
                    filenames[i] = NULL;
                }
                closedir(dp);
                return -2;
            }
            memcpy(name, entry->d_name, len + 1);
            filenames[count] = name;
        }
        count++;
    }

    closedir(dp);
    return count;
}

int32_t platform_read_file_to_buffer(const char *filepath, uint8_t **buffer, size_t *size) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        return -1;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }

    long file_size = ftell(fp);
    if (file_size < 0) {
        fclose(fp);
        return -1;
    }

    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return -1;
    }

    *buffer = (uint8_t *)platform_malloc(file_size);
    if (*buffer == NULL) {
        fclose(fp);
        return -1;
    }

    size_t bytes_read = fread(*buffer, 1, file_size, fp);
    fclose(fp);

    if (bytes_read != (size_t)file_size) {
        free(*buffer);
        *buffer = NULL;
        return -1;
    }

    *size = file_size;
    return 0;
}


// 将缓冲写入文件（不存在则创建，存在则截断）；成功 0，失败 -1
int32_t platform_write_buffer_to_file(const char *filepath, const uint8_t *buffer, size_t size) {
    if (!filepath || (!buffer && size > 0)) return -1;

    FILE *fp = fopen(filepath, "wb");
    if (!fp) return -1;

    size_t written = (size > 0) ? fwrite(buffer, 1, size, fp) : 0;
    fclose(fp);

    return (written == size) ? 0 : -1;
}

// 判断路径是否为目录：1 是；0 否或 stat 失败
int32_t platform_is_directory(const char *path) {
    if (!path) return 0;
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return S_ISDIR(st.st_mode) ? 1 : 0;
}

// ---------------- 随机访问文件读取（全局单句柄，与 ESP32 SD File 语义一致） ----------------

static FILE *s_platform_file = NULL;

int32_t platform_file_open(const char *filepath) {
    if (!filepath) return -1;
    platform_file_close(); // 同一时刻仅允许一个打开文件：先关闭旧句柄
    s_platform_file = fopen(filepath, "rb");
    return s_platform_file ? 0 : -1;
}

uint32_t platform_file_size(void) {
    if (!s_platform_file) return 0;
    long cur = ftell(s_platform_file);
    if (fseek(s_platform_file, 0, SEEK_END) != 0) return 0;
    long size = ftell(s_platform_file);
    if (cur >= 0) fseek(s_platform_file, cur, SEEK_SET); // 恢复原读写位置
    return (size > 0) ? (uint32_t)size : 0;
}

int32_t platform_file_seek(uint32_t offset) {
    if (!s_platform_file) return -1;
    return (fseek(s_platform_file, (long)offset, SEEK_SET) == 0) ? 0 : -1;
}

int32_t platform_file_read(uint8_t *buffer, size_t size) {
    if (!s_platform_file || !buffer) return -1;
    return (int32_t)fread(buffer, 1, size, s_platform_file); // 返回实际读取字节数
}

void platform_file_close(void) {
    if (s_platform_file) {
        fclose(s_platform_file);
        s_platform_file = NULL;
    }
}













void *platform_calloc(size_t nmemb, size_t size) {
    return calloc(nmemb, size);
}

void *platform_calloc_internal(size_t nmemb, size_t size) {
    return calloc(nmemb, size);
}

void *platform_malloc(size_t n) {
    return malloc(n);
}

void *platform_malloc_internal(size_t n) {
    return malloc(n);
}

void *platform_realloc(void *ptr, size_t n) {
    return realloc(ptr, n);
}

void *platform_realloc_internal(void *ptr, size_t n) {
    return realloc(ptr, n);
}

// Linux 上 malloc 直接由内核按需分配虚拟内存，无 ESP32 的堆碎片化问题；
// 此处以 sysinfo 报告的可用物理内存作为近似值。注意：返回值不可为 0，
// ui_animac.h 会依据 largest_free_block 是否超过 512K+64K 决定是否允许
// 创建编辑器内存池。

#define PLATFORM_HEAP_FALLBACK_BYTES (512u * 1024u * 1024u) // sysinfo 失败时的兜底值

static uint32_t platform_sys_free_bytes(void) {
    struct sysinfo info;
    if (sysinfo(&info) != 0) return PLATFORM_HEAP_FALLBACK_BYTES;
    uint64_t free_bytes = (uint64_t)info.freeram * (uint64_t)info.mem_unit;
    if (free_bytes > UINT32_MAX) free_bytes = UINT32_MAX;
    return (uint32_t)free_bytes;
}

uint32_t platform_get_free_heap_size() {
    return platform_sys_free_bytes();
}

uint32_t platform_get_largest_free_block() {
    // 虚拟内存下最大连续可分配块约等于可用内存
    return platform_sys_free_bytes();
}

// Linux 不区分 PSRAM / 内部 RAM，_internal 变体与外部一致
uint32_t platform_get_free_heap_size_internal() {
    return platform_sys_free_bytes();
}

uint32_t platform_get_largest_free_block_internal() {
    return platform_sys_free_bytes();
}






// ---------------- 任务抽象（pthread 实现，对应 ESP32 的 FreeRTOS） ----------------
//
// 说明：
// - 任务以 DETACHED 状态创建，入口函数返回即自动回收（业务代码约定入口
//   返回前调用 platform_task_delete_self，见 ui_ofdm.c）；
// - stack_bytes 语义与 xTaskCreate 一致（字节数），小于 PTHREAD_STACK_MIN
//   时提升到下限；
// - core >= 0 时尝试绑核（pthread_setaffinity_np），失败不视为错误；
// - priority 在 Linux 上无法在无特权情况下映射为实时调度优先级，故忽略；
// - 句柄直接承载 pthread_t 值（glibc 下为指针宽度整数）。

typedef struct {
    platform_task_func_t func;
    void *arg;
} Platform_Task_Bootstrap;

static void *platform_task_trampoline(void *p) {
    Platform_Task_Bootstrap *bootstrap = (Platform_Task_Bootstrap *)p;
    platform_task_func_t func = bootstrap->func;
    void *arg = bootstrap->arg;
    free(bootstrap);
    func(arg); // 业务约定：函数末尾调用 platform_task_delete_self()，不会返回
    return NULL;
}

int32_t platform_task_create(platform_task_func_t func, const char *name,
                             uint32_t stack_bytes, void *arg, int32_t priority,
                             int32_t core, platform_task_handle_t *out_handle) {
    (void)name;     // pthread 无需任务名
    (void)priority; // 无特权下无法设置实时优先级，忽略
    if (!func) return -1;

    Platform_Task_Bootstrap *bootstrap =
        (Platform_Task_Bootstrap *)calloc(1, sizeof(Platform_Task_Bootstrap));
    if (!bootstrap) return -1;
    bootstrap->func = func;
    bootstrap->arg  = arg;

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    if (stack_bytes >= (uint32_t)PTHREAD_STACK_MIN) {
        pthread_attr_setstacksize(&attr, (size_t)stack_bytes);
    }

    pthread_t tid;
    int err = pthread_create(&tid, &attr, platform_task_trampoline, bootstrap);
    pthread_attr_destroy(&attr);
    if (err != 0) {
        free(bootstrap);
        return -1;
    }

#ifdef __linux__
    if (core >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core, &cpuset);
        (void)pthread_setaffinity_np(tid, sizeof(cpuset), &cpuset); // 失败忽略
    }
#else
    (void)core;
#endif

    if (out_handle) {
        *out_handle = (platform_task_handle_t)tid;
    }
    return 0;
}

// 任务入口返回前必须调用（不返回）
void platform_task_delete_self(void) {
    pthread_exit(NULL);
}

// 强制删除任务（清理兜底；对应 vTaskDelete）
void platform_task_delete(platform_task_handle_t handle) {
    if (!handle) return;
    pthread_cancel((pthread_t)handle);
}

void platform_task_delay_ms(uint32_t ms) {
    usleep(ms * 1000);
}







// 普通 Linux 平台无振动马达，空操作
void set_vibration(uint32_t level) {
    (void)level;
}

// 主音量（0~255）：与 ui_app.c 中 global_state->volume 初值一致的默认值。
// 进程内静态保存；实际出声增益由 audio_out_linux.c 在 init/set_volume 时应用。
static uint8_t s_master_volume = 16;

void platform_set_master_volume(uint8_t volume) {
    s_master_volume = volume;
}

uint8_t platform_get_master_volume(void) {
    return s_master_volume;
}

