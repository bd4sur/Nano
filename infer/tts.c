#include "tts.h"
#include "platform.h"

#include <wchar.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define TTS_FIFO_PATH "/tmp/tts_fifo"
#define TTS_BUFFER_SIZE (4096)

// 传递给TTS的文字内容的命名管道
int g_tts_fifo_fd = 0;

// TTS分句用全局变量
int32_t g_tts_split_from = 0; // 切句子的起始位置


// 向TTS输入FIFO中写文本内容
int32_t write_tts_fifo(char *text_bytes, int32_t len) {
    // 如果没有fifo，创建fifo
    if (mkfifo(TTS_FIFO_PATH, 0666) == -1 && errno != EEXIST) {
        perror("tts fifo mkfifo failed");
        return -1;
    }
    // 以非阻塞写模式打开FIFO
    g_tts_fifo_fd = open(TTS_FIFO_PATH, O_WRONLY | O_NONBLOCK);
    if (g_tts_fifo_fd == -1) {
        perror("open tts fifo for writing failed");
        return -1;
    }

    ssize_t result = write(g_tts_fifo_fd, text_bytes, len);
    if (result == -1) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // FIFO缓冲区满，丢弃数据（不处理）
            // printf("FIFO full, data dropped\n");
        } else {
            perror("write failed");
            close(g_tts_fifo_fd);
            return -1;
        }
    } else {
        // 成功写入
        // printf("Wrote byte: %d\n", (unsigned char)data);
    }
    close(g_tts_fifo_fd);
    return 0;
}


int32_t send_tts_request(wchar_t *text, int32_t is_finished) {
    // TTS切句子
    wchar_t tts_chunk[TTS_BUFFER_SIZE];
    memset(tts_chunk, 0, TTS_BUFFER_SIZE * sizeof(wchar_t));
    int32_t wlen = wcslen(text);
    if (is_finished) {
        wcsncpy(tts_chunk, text + g_tts_split_from, wlen - g_tts_split_from);
        g_tts_split_from = 0;
    }
    else {
        for (int32_t i = g_tts_split_from; i < wlen; i++) {
            if (text[i] == L'，' ||
                text[i] == L'。' ||
                text[i] == L'\n' ||
                text[i] == L'：' ||
                text[i] == L'；' ||
                text[i] == L'？' ||
                text[i] == L'！') {
                if (i - g_tts_split_from > 6) {
                    wcsncpy(tts_chunk, text + g_tts_split_from, i + 1 - g_tts_split_from);
                    g_tts_split_from = i + 1;
                    break;
                }
            }
        }
    }

    // 非阻塞写FIFO
    char text_bytes[TTS_BUFFER_SIZE];
    memset(text_bytes, 0, TTS_BUFFER_SIZE);
    size_t len = wcstombs(text_bytes, tts_chunk, TTS_BUFFER_SIZE);
    if (len <= 0) {
        return -1;
    }
    printf("Write TTS FIFO: %s (%ld)\n", text_bytes, len);

    return write_tts_fifo(text_bytes, len);
}


int32_t stop_tts() {
    return write_tts_fifo("_TTS_STOP_", 11);
}

void reset_tts_split_status() {
    g_tts_split_from = 0;
}
