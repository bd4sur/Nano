#include "asr.h"
#include "platform.h"

#include <wchar.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define PTT_FIFO_PATH "/tmp/ptt_fifo"

#define ASR_FIFO_PATH "/tmp/asr_fifo"
#define ASR_BUFFER_SIZE (65536)

// 传递PTT状态的命名管道
static int g_ptt_fifo_fd = 0;

// 传递ASR识别结果的命名管道
static int g_asr_fifo_fd = 0;

// 向PTT状态FIFO中写PTT状态
int32_t set_ptt_status(uint8_t status) {
    if (mkfifo(PTT_FIFO_PATH, 0666) == -1 && errno != EEXIST) {
        perror("mkfifo failed");
        return -1;
    }
    // 以非阻塞写模式打开FIFO
    g_ptt_fifo_fd = open(PTT_FIFO_PATH, O_WRONLY | O_NONBLOCK);
    if (g_ptt_fifo_fd == -1) {
        perror("open fifo for writing failed");
        return -1;
    }
    // 尝试写入一个字节
    uint8_t data = status;
    ssize_t result = write(g_ptt_fifo_fd, &data, 1);
    if (result == -1) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // FIFO缓冲区满，丢弃数据（不处理）
            // printf("FIFO full, data dropped\n");
        } else {
            perror("write failed");
            return -1;
        }
    } else {
        // 成功写入
        // printf("Wrote byte: %d\n", (unsigned char)data);
    }
    return 0;
}

int32_t close_ptt_fifo() {
    return close(g_ptt_fifo_fd);
}



// 以只读方式打开ASR命名管道（非阻塞）
int32_t open_asr_fifo() {
    g_asr_fifo_fd = open(ASR_FIFO_PATH, O_RDONLY | O_NONBLOCK);
    if (g_asr_fifo_fd == -1) {
        perror("打开管道失败");
        return -1;
    }
    printf("管道打开成功，开始读取数据...\n");
    return 0;
}

int32_t close_asr_fifo() {
    return close(g_asr_fifo_fd);
}

// 读取ASR管道内容
int32_t read_asr_fifo(wchar_t *asr_text) {
    char asr_buffer[ASR_BUFFER_SIZE];
    memset(asr_buffer, 0, ASR_BUFFER_SIZE);

    ssize_t asr_bytes_read = read(g_asr_fifo_fd, asr_buffer, ASR_BUFFER_SIZE - 1);

    if (asr_bytes_read > 0) {
        asr_buffer[asr_bytes_read] = '\0';
        // printf("读取到数据: %s\n", asr_buffer); fflush(stdout);
        mbstowcs(asr_text, asr_buffer, ASR_BUFFER_SIZE);
    }
    else if (asr_bytes_read == 0) {
        // 管道写端关闭，重新打开
        printf("管道写端关闭，重新打开管道...\n");fflush(stdout);
        close(g_asr_fifo_fd);
        g_asr_fifo_fd = open(ASR_FIFO_PATH, O_RDONLY | O_NONBLOCK);
        if (g_asr_fifo_fd == -1) {
            // perror("重新打开管道失败");
            return -1;
        }
    }
    else {
        if (errno != EINTR) {
            // perror("读取管道失败");
        }
        return -1;
    }
    return (int32_t)asr_bytes_read;
}

// 穷人的ASR服务状态检测：通过读取ASR服务的日志前64kB中是否出现“init finished”来判断
int32_t check_asr_server_status() {
    char asr_log_buffer[65536];
    FILE *file = fopen(ASR_SERVER_LOG_PATH, "r");
    if (file == NULL) {
        return -1;
    }
    // 读取最多max_chars个字符
    size_t chars_read = fread(asr_log_buffer, sizeof(char), 65536 - 1, file);
    // 添加字符串结束符
    asr_log_buffer[chars_read] = '\0';
    fclose(file);
    // 查找日志中的模式
    char pattern[] = "asr model init finished. listen on port";
    // 使用strstr查找子字符串
    if (strstr(asr_log_buffer, pattern) != NULL) {
        return 1;
    }
    else {
        return 0;
    }
}

