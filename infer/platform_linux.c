#include "platform.h"

#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <locale.h>
#include <sys/stat.h>
#include <errno.h>

void sleep_in_ms(uint32_t ms) {
    usleep(ms * 1000);
}

// NOTE 返回的时间戳是32位的，存在2038问题！
uint32_t get_timestamp_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// 优雅关机
int32_t graceful_shutdown() {
    // 同步所有文件系统数据
    sync();
    // 等待同步完成
    sleep(2);
    // 执行关机
    if (system("sudo shutdown -h now") == -1) {
        perror("关机失败");
        return -1;
    }
    return 0;
}
