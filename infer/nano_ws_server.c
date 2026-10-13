// nano_ws_server.c —— Nano-Web 远程桌面服务器（见 nano_ws_server.h 的协议说明）
//
// 实现要点：
// - 零第三方依赖的 RFC 6455 子集：HTTP 升级握手（SHA-1 + Base64）、帧编解码、
//   Ping/Pong、Close；客户端帧按规范必须带掩码（兼容无掩码，宽容处理）。
// - 线程模型：1 个 accept 线程 + 每客户端 1 个连接线程（握手+收包）与 1 个发送线程。
//   发送经“全局帧缓冲镜像 + 每客户端脏区并集 + 条件变量”解耦：渲染线程只做
//   memcpy 与唤醒，发送节奏由客户端网络自适应（多条推送天然合并为一次脏区发送）。
// - SIGPIPE 全局忽略（send 到已关闭连接只返回 EPIPE，不杀进程）。

#define _GNU_SOURCE

#include "nano_ws_server.h"

#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

// TLS（HTTPS/WSS）：浏览器 DeviceMotion/DeviceOrientation 等传感器 API 要求安全上下文，
// 局域网 HTTP 页面上姿态事件不触发（iOS 连 requestPermission 都不存在），故服务端支持 TLS。
// 同端口协议嗅探双模：首字节 0x16（TLS ClientHello）→ SSL_accept；否则按明文 HTTP/WS 处理——
// 桌面端可继续用 http://（触屏可用），手机端用 https://（触屏+姿态可用）。
// 由 Makefile web 目标 -DNANO_WEB_TLS 启用；去掉该宏（及 -lssl -lcrypto）即回退纯明文。
#if defined(NANO_WEB_TLS)
#include <openssl/ssl.h>
#include <openssl/err.h>
#endif

// 构建期由 Makefile 从 web_remote_client.html 生成的嵌入页（兜底；磁盘可读时优先磁盘）
#if __has_include("web_client_html.h")
    #include "web_client_html.h"
#else
    static const char WEB_CLIENT_HTML[] =
        "<!DOCTYPE html><html><body><p>Nano Web Remote: web_remote_client.html not found.</p></body></html>\n";
#endif

// ------------------------------ 配置 ------------------------------

#define NANO_WS_MAX_CLIENTS        (8)       // 最大并发客户端数
#define NANO_WS_HTTP_REQ_MAX       (16384)   // HTTP 请求头最大字节数
#define NANO_WS_CLIENT_MSG_MAX     (4096)    // 客户端单条消息最大字节数（触屏消息仅 6 字节）
#define NANO_WS_HANDSHAKE_TIMEOUT_S (15)     // 握手阶段收包超时（秒）
#define NANO_WS_SEND_TIMEOUT_S     (5)       // 客户端 socket 发送超时（秒），防止僵死连接拖住发送线程

// ------------------------------ SHA-1（握手签名用） ------------------------------

typedef struct {
    uint32_t h[5];
    uint64_t total_len;
    uint8_t  block[64];
    size_t   block_len;
} Nano_SHA1;

static void sha1_process_block(Nano_SHA1 *ctx, const uint8_t *block) {
    uint32_t w[80];
    for (int32_t i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | (uint32_t)block[i * 4 + 3];
    }
    for (int32_t i = 16; i < 80; i++) {
        uint32_t t = w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16];
        w[i] = (t << 1) | (t >> 31);
    }
    uint32_t a = ctx->h[0], b = ctx->h[1], c = ctx->h[2], d = ctx->h[3], e = ctx->h[4];
    for (int32_t i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20)      { f = (b & c) | ((~b) & d);          k = 0x5A827999; }
        else if (i < 40) { f = b ^ c ^ d;                     k = 0x6ED9EBA1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d);   k = 0x8F1BBCDC; }
        else             { f = b ^ c ^ d;                     k = 0xCA62C1D6; }
        uint32_t tmp = ((a << 5) | (a >> 27)) + f + e + k + w[i];
        e = d; d = c; c = (b << 30) | (b >> 2); b = a; a = tmp;
    }
    ctx->h[0] += a; ctx->h[1] += b; ctx->h[2] += c; ctx->h[3] += d; ctx->h[4] += e;
}

static void sha1_init(Nano_SHA1 *ctx) {
    ctx->h[0] = 0x67452301; ctx->h[1] = 0xEFCDAB89; ctx->h[2] = 0x98BADCFE;
    ctx->h[3] = 0x10325476; ctx->h[4] = 0xC3D2E1F0;
    ctx->total_len = 0;
    ctx->block_len = 0;
}

static void sha1_update(Nano_SHA1 *ctx, const uint8_t *data, size_t len) {
    ctx->total_len += len;
    while (len > 0) {
        size_t n = 64 - ctx->block_len;
        if (n > len) n = len;
        memcpy(ctx->block + ctx->block_len, data, n);
        ctx->block_len += n;
        data += n;
        len -= n;
        if (ctx->block_len == 64) {
            sha1_process_block(ctx, ctx->block);
            ctx->block_len = 0;
        }
    }
}

static void sha1_final(Nano_SHA1 *ctx, uint8_t out[20]) {
    uint64_t bit_len = ctx->total_len * 8;
    uint8_t pad = 0x80;
    sha1_update(ctx, &pad, 1);
    uint8_t zero = 0;
    while (ctx->block_len != 56) sha1_update(ctx, &zero, 1);
    uint8_t len_be[8];
    for (int32_t i = 0; i < 8; i++) len_be[7 - i] = (uint8_t)(bit_len >> (i * 8));
    sha1_update(ctx, len_be, 8);
    for (int32_t i = 0; i < 5; i++) {
        out[i * 4]     = (uint8_t)(ctx->h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(ctx->h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(ctx->h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)(ctx->h[i]);
    }
}

// ------------------------------ Base64（握手签名用） ------------------------------

static const char B64_ALPHABET[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static void base64_encode(const uint8_t *in, size_t len, char *out) {
    size_t i = 0, o = 0;
    while (i + 3 <= len) {
        uint32_t v = ((uint32_t)in[i] << 16) | ((uint32_t)in[i + 1] << 8) | in[i + 2];
        out[o++] = B64_ALPHABET[(v >> 18) & 63];
        out[o++] = B64_ALPHABET[(v >> 12) & 63];
        out[o++] = B64_ALPHABET[(v >> 6) & 63];
        out[o++] = B64_ALPHABET[v & 63];
        i += 3;
    }
    if (i < len) {
        uint32_t v = (uint32_t)in[i] << 16;
        out[o++] = B64_ALPHABET[(v >> 18) & 63];
        if (i + 1 < len) {
            v |= (uint32_t)in[i + 1] << 8;
            out[o++] = B64_ALPHABET[(v >> 12) & 63];
            out[o++] = B64_ALPHABET[(v >> 6) & 63];
            out[o++] = '=';
        }
        else {
            out[o++] = B64_ALPHABET[(v >> 12) & 63];
            out[o++] = '=';
            out[o++] = '=';
        }
    }
    out[o] = '\0';
}

// ------------------------------ 连接 IO 抽象 ------------------------------

// 连接 IO：ssl==NULL 为明文（send/recv），否则走 TLS（SSL_write/SSL_read）
typedef struct {
    int32_t fd;
#if defined(NANO_WEB_TLS)
    SSL *ssl;
#endif
} Nano_WS_IO;

// ------------------------------ 客户端连接状态 ------------------------------

typedef struct Nano_WS_Client {
    Nano_WS_IO io;           // fd + TLS 上下文（ssl==NULL 明文）
    int32_t active;          // 1=连接存活（0 后发送线程退出、连接线程收尾）
    pthread_t sender_tid;
    pthread_cond_t cond;     // 脏区就绪信号（配 g_lock 使用）
    // 自上次发送以来未上屏的脏区并集 [x0,x1)x[y0,y1)；dirty_valid=0 表示无待发送内容
    int32_t dirty_x0, dirty_y0, dirty_x1, dirty_y1, dirty_valid;
    uint8_t *send_buf;       // 发送暂存（消息头 + 一帧最大像素量）
    size_t   send_buf_cap;
    uint16_t seq;            // 帧消息序号（客户端统计跳帧用）
    pthread_mutex_t send_lock; // 帧消息与 Pong 并发写同一 fd 的互斥
    struct Nano_WS_Client *next;
} Nano_WS_Client;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER; // 保护镜像/客户端链表/各客户端脏区
static Nano_WS_Client *g_clients = NULL;
static uint8_t  *g_mirror = NULL;   // 全局帧缓冲镜像（新客户端首帧的全屏来源）
static uint32_t  g_fb_w = 0, g_fb_h = 0;
static int32_t   g_listen_fd = -1;
static volatile int32_t g_running = 0;
static pthread_t g_accept_tid;

static char *g_index_page = NULL;  // 网页客户端页面（启动时磁盘读取，失败用嵌入兜底）
static size_t g_index_page_len = 0;

// 推送帧节奏（毫秒/帧）：发送线程按固定节奏取脏区并集发送，高频 push 自动合并，
// 消息率恒定（修复浏览器端事件循环积压导致的操作不跟手）。默认 60fps，
// 环境变量 NANO_WEB_MAX_FPS 可调（1~120）；空闲场景首帧即时发出，零额外延迟。
static uint32_t g_frame_budget_ms = 16;

// 单调毫秒时钟（帧节流用；CLOCK_MONOTONIC 不受系统时间调整影响）
static uint64_t nano_ws_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

#if defined(NANO_WEB_TLS)
static SSL_CTX *g_ssl_ctx = NULL;  // 非 NULL 时启用 TLS 嗅探（首字节 0x16 → SSL_accept）
#endif

// ------------------------------ socket 基础 ------------------------------

// 循环发送直至全部写完（MSG_NOSIGNAL 防 SIGPIPE；返回 0 成功）
static int32_t io_send_all(Nano_WS_IO *io, const uint8_t *buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n;
#if defined(NANO_WEB_TLS)
        if (io->ssl) {
            n = SSL_write(io->ssl, buf + sent, (int)(len - sent));
            if (n > 0) { sent += (size_t)n; continue; }
            int32_t err = SSL_get_error(io->ssl, (int)n);
            if (err == SSL_ERROR_SYSCALL && errno == EINTR) continue;
            return -1; // SSL_ERROR_ZERO_RETURN（对端关断）/SYSCALL（超时/复位）等：视为连接死亡
        }
#endif
        n = send(io->fd, buf + sent, len - sent, MSG_NOSIGNAL);
        if (n > 0) { sent += (size_t)n; continue; }
        if (n < 0 && errno == EINTR) continue;
        return -1; // EAGAIN（SO_SNDTIMEO 超时）/EPIPE/连接复位等：视为连接死亡
    }
    return 0;
}

// 循环接收直至读满（返回 0 成功；对端关闭/超时/错误返回 -1）
static int32_t io_recv_exact(Nano_WS_IO *io, uint8_t *buf, size_t len) {
    size_t got = 0;
    while (got < len) {
        ssize_t n;
#if defined(NANO_WEB_TLS)
        if (io->ssl) {
            n = SSL_read(io->ssl, buf + got, (int)(len - got));
            if (n > 0) { got += (size_t)n; continue; }
            int32_t err = SSL_get_error(io->ssl, (int)n);
            if (err == SSL_ERROR_SYSCALL && errno == EINTR) continue;
            return -1;
        }
#endif
        n = recv(io->fd, buf + got, len - got, 0);
        if (n > 0) { got += (size_t)n; continue; }
        if (n < 0 && errno == EINTR) continue;
        return -1;
    }
    return 0;
}

// ------------------------------ WebSocket 帧 ------------------------------

// 发送一个 WS 帧（FIN=1，服务端帧不掩码）。返回 0 成功。
static int32_t ws_send_frame(Nano_WS_Client *client, uint8_t opcode, const uint8_t *payload, size_t len) {
    uint8_t hdr[10];
    size_t hdr_len = 0;
    hdr[hdr_len++] = 0x80 | (opcode & 0x0F);
    if (len < 126) {
        hdr[hdr_len++] = (uint8_t)len;
    }
    else if (len <= 0xFFFF) {
        hdr[hdr_len++] = 126;
        hdr[hdr_len++] = (uint8_t)(len >> 8);
        hdr[hdr_len++] = (uint8_t)len;
    }
    else {
        hdr[hdr_len++] = 127;
        uint64_t v = len;
        for (int32_t i = 0; i < 8; i++) hdr[hdr_len++] = (uint8_t)(v >> ((7 - i) * 8));
    }
    pthread_mutex_lock(&client->send_lock);
    int32_t rc = io_send_all(&client->io, hdr, hdr_len);
    if (rc == 0 && len > 0) rc = io_send_all(&client->io, payload, len);
    pthread_mutex_unlock(&client->send_lock);
    return rc;
}

// ------------------------------ 客户端发送线程 ------------------------------

// 发送循环：等待脏区 → 从镜像拷出矩形像素并组帧 → 发送。
// 多条推送在等待期间自然合并为一次脏区并集（接收方带宽自适应降频）。
static void *nano_ws_sender_thread(void *arg) {
    Nano_WS_Client *client = (Nano_WS_Client *)arg;
    uint64_t next_tick = 0; // 下一帧最早发送时刻（帧节奏节流；0=立即）
    while (1) {
        pthread_mutex_lock(&g_lock);
        // 帧节奏节流（修复「消息洪泛导致浏览器积压、操作不跟手」）：
        //   - 无脏区：无限期等 signal（事件驱动场景零功耗等待）；
        //   - 有脏区但未到下帧时刻：睡到下帧时刻——期间渲染端的高频 push 只更新镜像并
        //     并入脏区并集，本线程以固定节奏（约 max_fps）取走最新状态发送，消息率恒定，
        //     浏览器端事件循环不再积压（此前 FLIP 场景每秒数千条消息导致严重滞后）；
        //   - 空闲后首帧：now >= next_tick 立即可发，菜单单击等单帧响应零额外延迟。
        while (client->active) {
            if (client->dirty_valid) {
                uint64_t now = nano_ws_now_ms();
                if (now >= next_tick) break;
                struct timespec ts;
                ts.tv_sec  = (time_t)(next_tick / 1000);
                ts.tv_nsec = (long)((next_tick % 1000) * 1000000L);
                pthread_cond_timedwait(&client->cond, &g_lock, &ts); // signal 提前醒则重判继续睡到点
            }
            else {
                pthread_cond_wait(&client->cond, &g_lock);
            }
        }
        if (!client->active) {
            pthread_mutex_unlock(&g_lock);
            break;
        }
        uint32_t x0 = (uint32_t)client->dirty_x0;
        uint32_t y0 = (uint32_t)client->dirty_y0;
        uint32_t w  = (uint32_t)(client->dirty_x1 - client->dirty_x0);
        uint32_t h  = (uint32_t)(client->dirty_y1 - client->dirty_y0);
        client->dirty_valid = 0;
        client->seq++;

        // 组帧：[0x03][seq u16][x][y][w][h]（u16 LE）+ RGBA8888 像素。
        // RGB888→RGBA 展开在服务端完成（浏览器端 img.data.set() 一次原生拷贝，
        // 消除 23 万像素的 JS 逐像素转换循环——跟手性的主要瓶颈之一）；
        // 从镜像展开时持有锁，保证不撕裂。
        size_t msg_len = 11 + (size_t)w * h * 4;
        if (msg_len > client->send_buf_cap) {
            // 镜像尺寸在启动时固定，正常不会走到这里（兜底扩容）
            uint8_t *nb = (uint8_t *)realloc(client->send_buf, msg_len);
            if (nb == NULL) { pthread_mutex_unlock(&g_lock); continue; }
            client->send_buf = nb;
            client->send_buf_cap = msg_len;
        }
        uint8_t *p = client->send_buf;
        p[0] = 0x03;
        p[1] = (uint8_t)(client->seq & 0xFF); p[2] = (uint8_t)(client->seq >> 8);
        p[3] = (uint8_t)(x0 & 0xFF);  p[4] = (uint8_t)(x0 >> 8);
        p[5] = (uint8_t)(y0 & 0xFF);  p[6] = (uint8_t)(y0 >> 8);
        p[7] = (uint8_t)(w & 0xFF);   p[8] = (uint8_t)(w >> 8);
        p[9] = (uint8_t)(h & 0xFF);   p[10] = (uint8_t)(h >> 8);
        uint8_t *dst = p + 11;
        for (uint32_t row = 0; row < h; row++) {
            const uint8_t *src = g_mirror + ((size_t)(y0 + row) * g_fb_w + x0) * 3;
            for (uint32_t col = 0; col < w; col++) {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = 255;
                src += 3;
                dst += 4;
            }
        }
        next_tick = nano_ws_now_ms() + g_frame_budget_ms;
        pthread_mutex_unlock(&g_lock);

        if (ws_send_frame(client, 0x2 /* binary */, client->send_buf, msg_len) != 0) {
            // 发送失败：关停本连接（shutdown 唤醒阻塞在 recv 的连接线程做收尾）
            pthread_mutex_lock(&g_lock);
            client->active = 0;
            pthread_mutex_unlock(&g_lock);
            shutdown(client->io.fd, SHUT_RDWR);
            break;
        }
    }
    return NULL;
}

// ------------------------------ 连接线程（握手 + 收包） ------------------------------

// 大小写不敏感的子串查找（HTTP 头解析用；hs 须为 NUL 结尾）
static const char *strcasestr_local(const char *hs, const char *needle) {
    size_t nl = strlen(needle);
    if (nl == 0) return hs;
    for (const char *p = hs; *p; p++) {
        size_t i = 0;
        while (i < nl && p[i] && tolower((uint8_t)p[i]) == tolower((uint8_t)needle[i])) i++;
        if (i == nl) return p;
    }
    return NULL;
}

// 读取 HTTP 请求头（直至 \r\n\r\n；上限 NANO_WS_HTTP_REQ_MAX）。返回读取字节数，失败 -1。
static int32_t http_read_request(Nano_WS_IO *io, char *buf, size_t cap) {
    size_t len = 0;
    while (len + 1 < cap) {
        uint8_t b;
        if (io_recv_exact(io, &b, 1) != 0) return -1;
        buf[len++] = (char)b;
        if (len >= 4 && memcmp(buf + len - 4, "\r\n\r\n", 4) == 0) break;
    }
    buf[len] = '\0';
    return (int32_t)len;
}

// 提供网页客户端页面（GET / 或 /index.html）
static void http_serve_index(Nano_WS_IO *io) {
    char hdr[256];
    int32_t hdr_len = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
        "Content-Length: %zu\r\nConnection: close\r\n\r\n", g_index_page_len);
    io_send_all(io, (const uint8_t *)hdr, (size_t)hdr_len);
    io_send_all(io, (const uint8_t *)g_index_page, g_index_page_len);
}

static void http_serve_404(Nano_WS_IO *io) {
    static const char resp[] = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    io_send_all(io, (const uint8_t *)resp, sizeof(resp) - 1);
}

// WebSocket 升级握手。成功返回 0。
static int32_t ws_handshake(Nano_WS_IO *io, const char *req) {
    const char *key_hdr = strcasestr_local(req, "Sec-WebSocket-Key:");
    if (key_hdr == NULL) return -1;
    const char *p = key_hdr + strlen("Sec-WebSocket-Key:");
    while (*p == ' ' || *p == '\t') p++;
    char key[128];
    size_t kl = 0;
    while (kl < sizeof(key) - 1 && p[kl] != '\r' && p[kl] != '\n' && p[kl] != '\0') {
        key[kl] = p[kl];
        kl++;
    }
    key[kl] = '\0';
    if (kl == 0) return -1;

    // accept = Base64(SHA1(key + GUID))
    static const char WS_GUID[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    Nano_SHA1 sha1;
    uint8_t digest[20];
    char accept_b64[32];
    sha1_init(&sha1);
    sha1_update(&sha1, (const uint8_t *)key, kl);
    sha1_update(&sha1, (const uint8_t *)WS_GUID, sizeof(WS_GUID) - 1);
    sha1_final(&sha1, digest);
    base64_encode(digest, sizeof(digest), accept_b64);

    char resp[256];
    int32_t resp_len = snprintf(resp, sizeof(resp),
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
        "Connection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n", accept_b64);
    return io_send_all(io, (const uint8_t *)resp, (size_t)resp_len);
}

// 处理一条完整的客户端消息（触屏/姿态事件等）
static void ws_on_client_message(const uint8_t *msg, size_t len) {
    if (len >= 6 && msg[0] == 0x10) { // 触屏事件
        int32_t action = msg[1];
        int32_t x = (int32_t)(int16_t)((uint16_t)msg[2] | ((uint16_t)msg[3] << 8));
        int32_t y = (int32_t)(int16_t)((uint16_t)msg[4] | ((uint16_t)msg[5] << 8));
        web_remote_on_touch(action, x, y);
    }
    else if (len >= 20 && msg[0] == 0x20) { // 姿态事件（f32/i16 小端）
        uint8_t flags = msg[1];
        float acc[3];
        float yaw_deg;
        int32_t orient_deg;
        memcpy(&acc[0], msg + 2, 4);
        memcpy(&acc[1], msg + 6, 4);
        memcpy(&acc[2], msg + 10, 4);
        memcpy(&yaw_deg, msg + 14, 4);
        orient_deg = (int32_t)(int16_t)((uint16_t)msg[18] | ((uint16_t)msg[19] << 8));
        web_remote_on_orientation((flags & 0x01) ? acc : NULL, yaw_deg, orient_deg, flags);
    }
}

// 收包循环：解 WS 帧、掩码还原、分片拼装；Close/Ping/Pong 处理。
// 返回时连接已不可用于接收（由调用方收尾）。
static void ws_recv_loop(Nano_WS_Client *client) {
    uint8_t msg[NANO_WS_CLIENT_MSG_MAX];
    size_t msg_len = 0;
    int32_t close_sent = 0;
    while (client->active) {
        uint8_t hdr[2];
        if (io_recv_exact(&client->io, hdr, 2) != 0) break;
        int32_t fin    = (hdr[0] & 0x80) != 0;
        int32_t opcode = hdr[0] & 0x0F;
        int32_t masked = (hdr[1] & 0x80) != 0;
        uint64_t len   = hdr[1] & 0x7F;
        if (len == 126) {
            uint8_t e[2];
            if (io_recv_exact(&client->io, e, 2) != 0) break;
            len = ((uint64_t)e[0] << 8) | e[1];
        }
        else if (len == 127) {
            uint8_t e[8];
            if (io_recv_exact(&client->io, e, 8) != 0) break;
            len = 0;
            for (int32_t i = 0; i < 8; i++) len = (len << 8) | e[i];
        }
        if (opcode < 0x8 && len > NANO_WS_CLIENT_MSG_MAX - msg_len) break; // 数据消息超长：协议异常，断开
        if (opcode >= 0x8 && len > 125) break; // 控制帧长度协议上限
        uint8_t mask[4] = {0, 0, 0, 0};
        if (masked && io_recv_exact(&client->io, mask, 4) != 0) break;
        uint8_t payload[256]; // 控制帧 ≤125 字节；数据帧按块流式拼装
        uint64_t remaining = len;
        uint32_t mask_off = 0;
        while (remaining > 0) {
            size_t chunk = remaining < sizeof(payload) ? (size_t)remaining : sizeof(payload);
            // 数据帧且不是首块（分片/大消息）才需要落 msg；控制帧直接在 payload 内处理
            if (io_recv_exact(&client->io, payload, chunk) != 0) goto out;
            for (size_t i = 0; i < chunk; i++) {
                payload[i] ^= mask[(mask_off + i) & 3];
            }
            mask_off = (mask_off + (uint32_t)chunk) & 3;
            if (opcode == 0x0 || opcode == 0x1 || opcode == 0x2) { // 数据帧（含分片续帧）
                memcpy(msg + msg_len, payload, chunk);
                msg_len += chunk;
            }
            else { // 控制帧：Ping/Pong/Close（长度 ≤125，单块读完）
                if (opcode == 0x9) { // Ping → Pong
                    ws_send_frame(client, 0xA, payload, chunk);
                }
                else if (opcode == 0x8) { // Close → 回 Close 后断开
                    if (!close_sent) {
                        ws_send_frame(client, 0x8, payload, chunk);
                        close_sent = 1;
                    }
                    goto out;
                }
            }
            remaining -= chunk;
        }
        if ((opcode == 0x0 || opcode == 0x1 || opcode == 0x2) && fin) {
            ws_on_client_message(msg, msg_len);
            msg_len = 0;
        }
    }
out:
    return;
}

// 连接线程：TLS 嗅探/握手 → HTTP 请求分发（静态页面 / WebSocket 升级）→ WS 连接全生命周期管理
static void *nano_ws_conn_thread(void *arg) {
    int32_t fd = (int32_t)(intptr_t)arg;
    Nano_WS_IO io = { .fd = fd };
#if defined(NANO_WEB_TLS)
    io.ssl = NULL;
#endif

    // 握手阶段收包超时（防慢速连接占住线程；对 SSL_accept 同样生效）
    struct timeval hs_tv = { .tv_sec = NANO_WS_HANDSHAKE_TIMEOUT_S, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &hs_tv, sizeof(hs_tv));

#if defined(NANO_WEB_TLS)
    // 协议嗅探：首字节 0x16（TLS Handshake）且 TLS 已启用 → SSL_accept；否则按明文处理
    if (g_ssl_ctx != NULL) {
        uint8_t first = 0;
        ssize_t n = recv(fd, &first, 1, MSG_PEEK);
        if (n == 1 && first == 0x16) {
            io.ssl = SSL_new(g_ssl_ctx);
            if (io.ssl == NULL || SSL_set_fd(io.ssl, fd) != 1 || SSL_accept(io.ssl) != 1) {
                if (io.ssl != NULL) SSL_free(io.ssl);
                close(fd);
                return NULL;
            }
        }
    }
#endif

    char *req = (char *)malloc(NANO_WS_HTTP_REQ_MAX);
    if (req == NULL) goto plain_close;
    if (http_read_request(&io, req, NANO_WS_HTTP_REQ_MAX) <= 0) {
        free(req);
        goto plain_close;
    }

    // 解析请求行（仅支持 GET）
    char path[256] = {0};
    if (sscanf(req, "GET %255s", path) != 1) {
        http_serve_404(&io);
        free(req);
        goto plain_close;
    }

    int32_t is_ws = (strcmp(path, "/ws") == 0) &&
                    strcasestr_local(req, "Upgrade:") != NULL &&
                    strcasestr_local(req, "websocket") != NULL &&
                    strcasestr_local(req, "Sec-WebSocket-Key:") != NULL;

    if (!is_ws) {
        if (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0) http_serve_index(&io);
        else http_serve_404(&io);
        free(req);
        goto plain_close;
    }

    if (ws_handshake(&io, req) != 0) {
        free(req);
        goto plain_close;
    }
    free(req);

    // 客户端数上限
    pthread_mutex_lock(&g_lock);
    int32_t n_clients = 0;
    for (Nano_WS_Client *c = g_clients; c != NULL; c = c->next) n_clients++;
    pthread_mutex_unlock(&g_lock);
    if (n_clients >= NANO_WS_MAX_CLIENTS) {
        static const char rej[] = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n";
        io_send_all(&io, (const uint8_t *)rej, sizeof(rej) - 1);
        goto plain_close;
    }

    // 注册客户端
    Nano_WS_Client *client = (Nano_WS_Client *)calloc(1, sizeof(Nano_WS_Client));
    if (client == NULL) goto plain_close;
    client->io = io; // TLS 上下文移交客户端（本线程后续经 client->io 收发）
    client->active = 1;
    // 条件变量使用单调时钟（与发送线程节流的 nano_ws_now_ms 同一基准）
    pthread_condattr_t cond_attr;
    pthread_condattr_init(&cond_attr);
    pthread_condattr_setclock(&cond_attr, CLOCK_MONOTONIC);
    pthread_cond_init(&client->cond, &cond_attr);
    pthread_condattr_destroy(&cond_attr);
    pthread_mutex_init(&client->send_lock, NULL);
    client->send_buf_cap = 9 + (size_t)g_fb_w * g_fb_h * 3;
    client->send_buf = (uint8_t *)malloc(client->send_buf_cap);
    if (client->send_buf == NULL) {
        pthread_cond_destroy(&client->cond);
        pthread_mutex_destroy(&client->send_lock);
        free(client);
        goto plain_close;
    }

    // WS 数据阶段：关闭接收超时（长连接），开启 TCP_NODELAY 降低触屏事件延迟；
    // 发送超时兜底僵死连接
    struct timeval tv0 = { .tv_sec = 0, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv0, sizeof(tv0));
    struct timeval stv = { .tv_sec = NANO_WS_SEND_TIMEOUT_S, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &stv, sizeof(stv));
    int32_t one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    pthread_mutex_lock(&g_lock);
    client->next = g_clients;
    g_clients = client;
    // 新客户端首帧为全屏（镜像可能尚未有任何推送，为零黑屏；业务首帧到达后即自愈）
    client->dirty_x0 = 0;
    client->dirty_y0 = 0;
    client->dirty_x1 = (int32_t)g_fb_w;
    client->dirty_y1 = (int32_t)g_fb_h;
    client->dirty_valid = 1;
    pthread_mutex_unlock(&g_lock);

    // 初始化消息（帧缓冲尺寸），随后启动发送线程
    uint8_t init_msg[5] = {
        0x00,
        (uint8_t)(g_fb_w & 0xFF), (uint8_t)(g_fb_w >> 8),
        (uint8_t)(g_fb_h & 0xFF), (uint8_t)(g_fb_h >> 8)
    };
    if (ws_send_frame(client, 0x2, init_msg, sizeof(init_msg)) != 0) goto cleanup;

    if (pthread_create(&client->sender_tid, NULL, nano_ws_sender_thread, client) != 0) goto cleanup;

    printf("nano-web: client connected (fd=%d%s)\n", fd, client->io.ssl ? ", TLS" : "");
    ws_recv_loop(client); // 阻塞至连接断开
    printf("nano-web: client disconnected (fd=%d)\n", fd);

cleanup:
    // 停发送线程（唤醒其条件变量后汇合）
    pthread_mutex_lock(&g_lock);
    client->active = 0;
    pthread_cond_signal(&client->cond);
    pthread_mutex_unlock(&g_lock);
    shutdown(fd, SHUT_RDWR);
    if (client->sender_tid) pthread_join(client->sender_tid, NULL);

    // 摘链、释放
    pthread_mutex_lock(&g_lock);
    Nano_WS_Client **pp = &g_clients;
    while (*pp != NULL && *pp != client) pp = &(*pp)->next;
    if (*pp == client) *pp = client->next;
    pthread_mutex_unlock(&g_lock);

    free(client->send_buf);
    pthread_cond_destroy(&client->cond);
    pthread_mutex_destroy(&client->send_lock);
    io = client->io; // TLS 上下文从 client 取回（free 之前），由下方统一关闭
    free(client);
    // fallthrough：统一关闭 TLS 与 fd
plain_close:
#if defined(NANO_WEB_TLS)
    if (io.ssl != NULL) {
        SSL_shutdown(io.ssl); // 对端已死时靠 SO_SNDTIMEO 兜底，最坏 5s
        SSL_free(io.ssl);
    }
#endif
    close(fd);
    return NULL;
}

// ------------------------------ accept 线程 ------------------------------

static void *nano_ws_accept_thread(void *arg) {
    (void)arg;
    while (g_running) {
        int32_t fd = accept(g_listen_fd, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            break; // 监听 fd 已关闭（stop）或不可恢复错误
        }
        pthread_t tid;
        if (pthread_create(&tid, NULL, nano_ws_conn_thread, (void *)(intptr_t)fd) != 0) {
            close(fd);
            continue;
        }
        pthread_detach(tid);
    }
    return NULL;
}

// ------------------------------ 对外接口 ------------------------------

#if defined(NANO_WEB_TLS)
static int32_t nano_ws_tls_load(const char *cert_path, const char *key_path) {
    SSL_CTX *ctx = SSL_CTX_new(TLS_server_method());
    if (ctx == NULL) return -1;
    SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION); // 禁用 TLS1.0/1.1
    if (SSL_CTX_use_certificate_chain_file(ctx, cert_path) != 1 ||
        SSL_CTX_use_PrivateKey_file(ctx, key_path, SSL_FILETYPE_PEM) != 1 ||
        SSL_CTX_check_private_key(ctx) != 1) {
        SSL_CTX_free(ctx);
        return -1;
    }
    g_ssl_ctx = ctx;
    printf("nano-web: TLS enabled (cert=%s)\n", cert_path);
    return 0;
}

// 加载 TLS 证书/私钥（自签名证书由 tools/gen_web_cert.sh 生成）。成功返回 0（g_ssl_ctx 就绪），
// 无证书文件/加载失败返回 -1（服务器以明文 HTTP/WS 运行，行为同未启用 TLS）。
static int32_t nano_ws_tls_init(void) {
    const char *env_cert = getenv("NANO_WEB_TLS_CERT");
    const char *env_key  = getenv("NANO_WEB_TLS_KEY");
    if (env_cert != NULL && env_key != NULL) {
        return nano_ws_tls_load(env_cert, env_key);
    }
    // 候选路径（与 index 页面同一思路：exe 相对位置与 cwd）
    char exe_dir[1024] = "";
    ssize_t n = readlink("/proc/self/exe", exe_dir, sizeof(exe_dir) - 1);
    if (n > 0) {
        exe_dir[n] = '\0';
        char *slash = strrchr(exe_dir, '/');
        if (slash != NULL) *slash = '\0'; // exe_dir 现为可执行文件所在目录（bin/）
    }
    char c1[1200], k1[1200];
    snprintf(c1, sizeof(c1), "%s/web_cert.pem", exe_dir);   // bin/web_cert.pem（exe 在 bin/）
    snprintf(k1, sizeof(k1), "%s/web_key.pem", exe_dir);
    const char *cert_pairs[][2] = {
        {c1, k1},
        {"bin/web_cert.pem", "bin/web_key.pem"},            // cwd=infer/
        {NULL, NULL}
    };
    for (int32_t i = 0; cert_pairs[i][0] != NULL; i++) {
        if (access(cert_pairs[i][0], R_OK) == 0 && access(cert_pairs[i][1], R_OK) == 0) {
            return nano_ws_tls_load(cert_pairs[i][0], cert_pairs[i][1]);
        }
    }
    return -1;
}
#endif

// 启动时加载网页客户端页面：磁盘优先（便于免重编译调前端），嵌入页兜底
static void nano_ws_load_index_page(void) {
    char exe_path[1024];
    char exe_dir_candidate[1200];
    exe_dir_candidate[0] = '\0';
    ssize_t n = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (n > 0) {
        exe_path[n] = '\0';
        char *slash = strrchr(exe_path, '/');
        if (slash != NULL) {
            *slash = '\0';
            snprintf(exe_dir_candidate, sizeof(exe_dir_candidate),
                     "%s/../web_remote_client.html", exe_path);
        }
    }
    const char *candidates[] = {
        "web_remote_client.html",        // 工作目录=infer/
        "infer/web_remote_client.html",  // 工作目录=仓库根
        exe_dir_candidate,               // 相对可执行文件位置（bin/../）
        NULL
    };
    for (int32_t i = 0; candidates[i] != NULL && candidates[i][0] != '\0'; i++) {
        FILE *fp = fopen(candidates[i], "rb");
        if (fp == NULL) continue;
        fseek(fp, 0, SEEK_END);
        long size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        if (size > 0 && size < 4 * 1024 * 1024) {
            g_index_page = (char *)malloc((size_t)size);
            if (g_index_page != NULL && fread(g_index_page, 1, (size_t)size, fp) == (size_t)size) {
                g_index_page_len = (size_t)size;
                printf("nano-web: client page loaded from %s (%zu bytes)\n", candidates[i], g_index_page_len);
                fclose(fp);
                return;
            }
            free(g_index_page);
            g_index_page = NULL;
        }
        fclose(fp);
    }
    g_index_page = (char *)WEB_CLIENT_HTML;
    g_index_page_len = sizeof(WEB_CLIENT_HTML) - 1;
    printf("nano-web: client page: using embedded copy (%zu bytes)\n", g_index_page_len);
}

int32_t nano_ws_server_start(uint16_t port, uint32_t fb_width, uint32_t fb_height) {
    if (g_running) nano_ws_server_stop();

    // send 到已关闭连接返回 EPIPE 而非 SIGPIPE 杀进程（全进程生效）
    signal(SIGPIPE, SIG_IGN);

#if defined(NANO_WEB_TLS)
    if (nano_ws_tls_init() != 0) {
        printf("nano-web: no TLS certificate found (bin/web_cert.pem + web_key.pem), "
               "running in plain HTTP/WS mode.\n"
               "         HTTPS/WSS is required for browser motion sensors on phones "
               "(run tools/gen_web_cert.sh to generate a self-signed certificate).\n");
    }
#endif

    g_fb_w = fb_width;
    g_fb_h = fb_height;

    // 推送帧节奏（默认 60fps；NANO_WEB_MAX_FPS 可覆盖，钳制 1~120）
    const char *fps_env = getenv("NANO_WEB_MAX_FPS");
    if (fps_env != NULL && atoi(fps_env) > 0) {
        int32_t fps = atoi(fps_env);
        if (fps > 120) fps = 120;
        g_frame_budget_ms = (uint32_t)(1000 / fps);
        if (g_frame_budget_ms == 0) g_frame_budget_ms = 1;
    }
    free(g_mirror);
    g_mirror = (uint8_t *)calloc((size_t)fb_width * fb_height * 3, 1);
    if (g_mirror == NULL) return -1;

    nano_ws_load_index_page();

    g_listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_listen_fd < 0) return -2;
    int32_t one = 1;
    setsockopt(g_listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY); // 0.0.0.0
    addr.sin_port = htons(port);
    if (bind(g_listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(g_listen_fd);
        g_listen_fd = -1;
        return -3;
    }
    if (listen(g_listen_fd, NANO_WS_MAX_CLIENTS) != 0) {
        close(g_listen_fd);
        g_listen_fd = -1;
        return -4;
    }

    g_running = 1;
    if (pthread_create(&g_accept_tid, NULL, nano_ws_accept_thread, NULL) != 0) {
        close(g_listen_fd);
        g_listen_fd = -1;
        g_running = 0;
        return -5;
    }
    pthread_detach(g_accept_tid);
    return 0;
}

void nano_ws_server_stop(void) {
    g_running = 0;
    if (g_listen_fd >= 0) {
        shutdown(g_listen_fd, SHUT_RDWR);
        close(g_listen_fd);
        g_listen_fd = -1;
    }
    pthread_mutex_lock(&g_lock);
    for (Nano_WS_Client *c = g_clients; c != NULL; c = c->next) {
        c->active = 0;
        pthread_cond_signal(&c->cond);
        shutdown(c->io.fd, SHUT_RDWR);
    }
    pthread_mutex_unlock(&g_lock);
    // 连接线程自行摘链释放；进程退出场景下不再逐一会合
}

int32_t nano_ws_server_client_count(void) {
    pthread_mutex_lock(&g_lock);
    int32_t n = 0;
    for (Nano_WS_Client *c = g_clients; c != NULL; c = c->next) n++;
    pthread_mutex_unlock(&g_lock);
    return n;
}

void nano_ws_server_push_frame(const uint8_t *rgb888, uint32_t fb_width, uint32_t fb_height,
                               uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    if (!g_running || g_mirror == NULL) return;
    if (fb_width != g_fb_w || fb_height != g_fb_h) return; // 尺寸不匹配（不会发生：启动时已固定）
    if (x >= fb_width || y >= fb_height) return;
    if (x + w > fb_width)  w = fb_width - x;
    if (y + h > fb_height) h = fb_height - y;
    if (w == 0 || h == 0) return;

    pthread_mutex_lock(&g_lock);
    // 并入全局镜像（即使暂无客户端也要更新：新客户端首帧以镜像全屏推送）
    for (uint32_t row = 0; row < h; row++) {
        memcpy(g_mirror + ((size_t)(y + row) * g_fb_w + x) * 3,
               rgb888 + ((size_t)(y + row) * fb_width + x) * 3,
               (size_t)w * 3);
    }
    // 各客户端脏区并集 + 唤醒发送线程
    int32_t x1 = (int32_t)(x + w), y1 = (int32_t)(y + h);
    for (Nano_WS_Client *c = g_clients; c != NULL; c = c->next) {
        if (!c->dirty_valid) {
            c->dirty_x0 = (int32_t)x; c->dirty_y0 = (int32_t)y;
            c->dirty_x1 = x1;         c->dirty_y1 = y1;
            c->dirty_valid = 1;
        }
        else {
            if ((int32_t)x < c->dirty_x0) c->dirty_x0 = (int32_t)x;
            if ((int32_t)y < c->dirty_y0) c->dirty_y0 = (int32_t)y;
            if (x1 > c->dirty_x1) c->dirty_x1 = x1;
            if (y1 > c->dirty_y1) c->dirty_y1 = y1;
        }
        pthread_cond_signal(&c->cond);
    }
    pthread_mutex_unlock(&g_lock);
}
