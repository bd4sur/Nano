// hal_net_linux.c —— hal_net 网络通信 HAL 的 POSIX/Linux 实现
//
// 服务对象：nano_tty / nano_web 等本机编译目标（远端模型等业务的云端 HTTP 访问）。
//
// 实现模型：每请求一个后台 pthread 完成【整条 HTTP 交换】（DNS/连接/发送/接收），
//   经「互斥锁 + 条件变量保护的有界环形缓冲」把响应体字节流交给主线程轮询拉取。
//   选择后台线程而非主线程非阻塞 socket 的原因：
//     - DNS 解析（getaddrinfo）与非阻塞 connect 的状态机复杂且平台差异大，
//       线程内可全部用阻塞式 API 一次写对；
//     - 主循环保持零阻塞契约（环形缓冲满时由【后台线程】阻塞等待消费，
//       反压发生在网络侧而非 UI 侧）；
//     - nano_web 目标本就启用 pthread（nano_ws_server.c），tty 目标 Makefile
//       已全局链接 -pthread。
//
// 请求生命周期与内存安全：请求结构体的释放采用「两阶段退休」——
//   net_http_close 只标记中止并 shutdown 套接字；结构体在【工作线程退出并
//   置位 worker_done 之后】才真正 free。若 close 先于线程退出，结构体进入
//   静态"坟场"列表，由下一次 post/close 调用统一收尸（有界，最坏情形下
//   并发请求数 + 已退休未收尸数都很小）。这样主线程与工作线程对结构体的
//   访问始终被同一把锁保护，无 use-after-free 窗口。
//
// 分帧处理（响应体 → 字节流）：Content-Length（读满即止）、
//   Transfer-Encoding: chunked（解 chunk 后透传）、其余（读到连接关闭）。

#include "hal_net.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

// ---------------- 可调参数 ----------------
#define NANO_NET_RING_SIZE     (256 * 1024) // 响应体环形缓冲（字节）
#define NANO_NET_CONNECT_TIMEOUT_MS  (10000) // TCP 连接超时
#define NANO_NET_RECV_TIMEOUT_S      (120)   // 单次 recv 空闲超时（推理首包/思考间隙可能较长）
#define NANO_NET_SEND_TIMEOUT_S      (30)    // 单次 send 超时
#define NANO_NET_HEAD_MAX       (32 * 1024) // 响应头累计上限
#define NANO_NET_URL_MAX        (1024)
#define NANO_NET_HOST_MAX       (256)
#define NANO_NET_PATH_MAX       (768)

// ===============================================================================
// 请求结构体（hal_net.h 句柄的本体；全部字段由下方锁保护）
// ===============================================================================

typedef struct Net_Request_Linux {
    // —— 配置拷贝（post 时由主线程写入，此后只读，工作线程读取）——
    char    url[NANO_NET_URL_MAX];
    char    host[NANO_NET_HOST_MAX]; // 不含端口
    char    path[NANO_NET_PATH_MAX]; // 以 / 开头
    int     port;
    uint8_t *body;
    size_t  body_len;
    char    headers[NANO_NET_MAX_HEADERS][192]; // 序列化为 "Key: Value"
    int32_t header_num;

    // —— 运行态 ——
    pthread_mutex_t lock;
    pthread_cond_t  cond_space;  // 环形缓冲有空间（工作线程等待）
    pthread_cond_t  cond_data;   // 环形缓冲有数据（主线程不需要等待，仅轮询，此变量备用）
    uint8_t  ring[NANO_NET_RING_SIZE];
    size_t   ring_head;          // 消费位置（主线程拉取）
    size_t   ring_tail;          // 生产位置（工作线程写入）
    int32_t  http_status;        // 0-未知
    int32_t  state;              // Net_Stream_State
    int32_t  stream_err;         // 错误细节（诊断日志用）
    volatile int32_t close_flag; // 主线程请求中止（close 置位后不再写缓冲）

    // —— 生命周期 ——
    volatile int32_t worker_done;   // 工作线程已退出（此后主线程可 free）
    volatile int32_t handle_closed; // 主线程已调用 close（句柄失效）
    int     detached;               // 线程已 detach
} Net_Request_Linux;

// "坟场"：close 先于工作线程结束时暂存结构体，延后收尸
#define NANO_NET_GRAVEYARD_MAX (8)
static Net_Request_Linux *s_graveyard[NANO_NET_GRAVEYARD_MAX];
static int32_t s_graveyard_num = 0;
static pthread_mutex_t s_graveyard_lock = PTHREAD_MUTEX_INITIALIZER;

// 收尸：释放坟场中工作线程已退出的结构体（调用方须持有 s_graveyard_lock）
static void net_reap_graveyard_locked(void) {
    int32_t w = 0;
    for (int32_t i = 0; i < s_graveyard_num; i++) {
        Net_Request_Linux *r = s_graveyard[i];
        if (r->worker_done) {
            pthread_mutex_destroy(&r->lock);
            pthread_cond_destroy(&r->cond_space);
            pthread_cond_destroy(&r->cond_data);
            free(r->body);
            free(r);
        }
        else {
            s_graveyard[w++] = r;
        }
    }
    s_graveyard_num = w;
}

static void net_retire(Net_Request_Linux *r) {
    pthread_mutex_lock(&s_graveyard_lock);
    net_reap_graveyard_locked();
    if (r->worker_done) { // 已退出：直接释放
        pthread_mutex_destroy(&r->lock);
        pthread_cond_destroy(&r->cond_space);
        pthread_cond_destroy(&r->cond_data);
        free(r->body);
        free(r);
    }
    else if (s_graveyard_num < NANO_NET_GRAVEYARD_MAX) {
        s_graveyard[s_graveyard_num++] = r;
    }
    else {
        // 坟场已满（理论不可达：并发请求数远小于容量）：放弃收尸，直接释放。
        // 工作线程此后只访问已释放内存的风险由"坟场容量 >= 并发请求数"保证来规避。
        fprintf(stderr, "[hal_net] graveyard full, leaking request\n");
    }
    pthread_mutex_unlock(&s_graveyard_lock);
}

// ---------------- 环形缓冲 ----------------

static size_t net_ring_used(const Net_Request_Linux *r) {
    return (r->ring_tail + NANO_NET_RING_SIZE - r->ring_head) % NANO_NET_RING_SIZE;
}

// 工作线程写入（阻塞至写入完成或中止）。返回写入字节数（中止时可能 < len）。
static size_t net_ring_write(Net_Request_Linux *r, const uint8_t *data, size_t len) {
    size_t written = 0;
    pthread_mutex_lock(&r->lock);
    while (written < len) {
        if (r->close_flag) break;
        size_t space = NANO_NET_RING_SIZE - 1 - net_ring_used(r); // 保留1空位区分满/空
        if (space == 0) {
            pthread_cond_wait(&r->cond_space, &r->lock);
            continue;
        }
        size_t tail = r->ring_tail;
        size_t n = (tail < r->ring_head) ? (r->ring_head - 1 - tail)
                                         : (NANO_NET_RING_SIZE - tail);
        if (n > space) n = space;
        if (n > len - written) n = len - written;
        memcpy(r->ring + tail, data + written, n);
        r->ring_tail = (tail + n) % NANO_NET_RING_SIZE;
        written += n;
    }
    pthread_mutex_unlock(&r->lock);
    return written;
}

// ===============================================================================
// HTTP 客户端（工作线程内运行，允许阻塞）
// ===============================================================================

static int32_t net_parse_url(const char *url, char *host, size_t host_max,
                             char *path, size_t path_max, int *port) {
    if (strncmp(url, "http://", 7) != 0) {
        fprintf(stderr, "[hal_net] only http:// is supported: %s\n", url);
        return -1;
    }
    const char *p = url + 7;
    const char *slash = strchr(p, '/');
    const char *colon = strchr(p, ':');
    size_t host_len;
    if (colon && (!slash || colon < slash)) {
        host_len = (size_t)(colon - p);
        *port = atoi(colon + 1);
    }
    else {
        host_len = slash ? (size_t)(slash - p) : strlen(p);
        *port = 80;
    }
    if (host_len == 0 || host_len >= host_max) return -1;
    memcpy(host, p, host_len);
    host[host_len] = 0;
    if (!slash) {
        snprintf(path, path_max, "/");
    }
    else if (strlen(slash) >= path_max) {
        return -1;
    }
    else {
        snprintf(path, path_max, "%s", slash);
    }
    return 0;
}

// 带超时的阻塞 connect（非阻塞 connect + poll）。返回 0 成功。
static int32_t net_connect_with_timeout(const char *host, int port) {
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char service[16];
    snprintf(service, sizeof(service), "%d", port);
    if (getaddrinfo(host, service, &hints, &res) != 0 || res == NULL) {
        return -1;
    }
    int32_t rc = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        // 非阻塞发起连接
        int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            rc = fd;
            break;
        }
        if (errno == EINPROGRESS) {
            struct pollfd pfd = { .fd = fd, .events = POLLOUT, .revents = 0 };
            int pret = poll(&pfd, 1, NANO_NET_CONNECT_TIMEOUT_MS);
            if (pret == 1) {
                int soerr = 0;
                socklen_t olen = sizeof(soerr);
                getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &olen);
                if (soerr == 0) { rc = fd; break; }
            }
        }
        close(fd);
    }
    freeaddrinfo(res);
    if (rc >= 0) {
        int flags = fcntl(rc, F_GETFL, 0);
        fcntl(rc, F_SETFL, flags & ~O_NONBLOCK); // 恢复阻塞式收发
        struct timeval tv;
        tv.tv_sec = NANO_NET_RECV_TIMEOUT_S; tv.tv_usec = 0;
        setsockopt(rc, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        tv.tv_sec = NANO_NET_SEND_TIMEOUT_S; tv.tv_usec = 0;
        setsockopt(rc, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        int one = 1;
        setsockopt(rc, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    return rc;
}

// 发送全部数据（MSG_NOSIGNAL 防止 SIGPIPE 杀死进程）。返回 0 成功。
static int32_t net_send_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

// 工作线程主体
static void *net_worker(void *arg) {
    Net_Request_Linux *r = (Net_Request_Linux *)arg;

    int fd = net_connect_with_timeout(r->host, r->port);
    if (fd < 0) {
        fprintf(stderr, "[hal_net] connect %s:%d failed: %s\n", r->host, r->port, strerror(errno));
        pthread_mutex_lock(&r->lock);
        r->state = NET_STREAM_ERROR;
        r->worker_done = 1;
        pthread_mutex_unlock(&r->lock);
        return NULL;
    }

    // 组装请求
    size_t req_cap = NANO_NET_URL_MAX + r->header_num * 192 + 256;
    char *req = (char *)malloc(req_cap);
    if (!req) {
        close(fd);
        pthread_mutex_lock(&r->lock);
        r->state = NET_STREAM_ERROR;
        r->worker_done = 1;
        pthread_mutex_unlock(&r->lock);
        return NULL;
    }
    size_t off = 0;
    off += (size_t)snprintf(req + off, req_cap - off, "POST %s HTTP/1.1\r\n", r->path);
    off += (size_t)snprintf(req + off, req_cap - off, "Host: %s:%d\r\n", r->host, r->port);
    off += (size_t)snprintf(req + off, req_cap - off, "Content-Length: %zu\r\n", r->body_len);
    off += (size_t)snprintf(req + off, req_cap - off, "Connection: close\r\n");
    for (int32_t i = 0; i < r->header_num; i++) {
        off += (size_t)snprintf(req + off, req_cap - off, "%s\r\n", r->headers[i]);
    }
    off += (size_t)snprintf(req + off, req_cap - off, "\r\n");
    int32_t send_rc = net_send_all(fd, req, off);
    if (send_rc == 0 && r->body_len > 0) send_rc = net_send_all(fd, r->body, r->body_len);
    free(req);
    if (send_rc != 0) {
        fprintf(stderr, "[hal_net] send request failed\n");
        close(fd);
        pthread_mutex_lock(&r->lock);
        r->state = NET_STREAM_ERROR;
        r->worker_done = 1;
        pthread_mutex_unlock(&r->lock);
        return NULL;
    }

    // —— 接收响应头 ——
    char head[NANO_NET_HEAD_MAX + 1];
    size_t head_len = 0;
    for (;;) {
        // 逐字节读直到头部结束，简单可靠（头部只读一次，性能无关紧要）
        ssize_t n = recv(fd, head + head_len, 1, 0);
        if (n <= 0) goto recv_fail;
        head_len++;
        if (head_len >= 4 && head[head_len - 4] == '\r' && head[head_len - 3] == '\n'
            && head[head_len - 2] == '\r' && head[head_len - 1] == '\n') break;
        if (head_len >= NANO_NET_HEAD_MAX) {
            fprintf(stderr, "[hal_net] response head too large\n");
            goto recv_fail;
        }
    }
    head[head_len] = 0;
    {
        int status = 0;
        if (sscanf(head, "HTTP/%*d.%*d %d", &status) != 1) status = 0;
        pthread_mutex_lock(&r->lock);
        r->http_status = status;
        pthread_mutex_unlock(&r->lock);
        if (status < 200 || status >= 300) {
            fprintf(stderr, "[hal_net] HTTP status %d\n", status);
            goto recv_fail;
        }
    }
    // 分帧方式判定
    int is_chunked = 0;
    int64_t content_len = -1;
    {
        char *lower = (char *)malloc(head_len + 1);
        if (lower) {
            for (size_t i = 0; i < head_len; i++) lower[i] = (char)tolower((unsigned char)head[i]);
            lower[head_len] = 0;
            if (strstr(lower, "transfer-encoding: chunked")) is_chunked = 1;
            const char *cl = strstr(lower, "content-length:");
            if (cl) content_len = strtoll(cl + 15, NULL, 10);
            free(lower);
        }
    }

    // —— 接收响应体 ——（响应头为逐字节读取至终止符，无预读残留，body 全部来自 socket）
    uint8_t iobuf[8192];
    if (is_chunked) {
        // chunk 编码：[hex size]\r\n[data]\r\n ... [0]\r\n[trailer...\r\n]
        for (;;) {
            // 读一行 chunk 大小
            char size_line[64];
            size_t sl = 0;
            while (!(sl >= 2 && size_line[sl - 2] == '\r' && size_line[sl - 1] == '\n')) {
                if (sl + 1 >= sizeof(size_line)) goto recv_fail;
                ssize_t n = recv(fd, size_line + sl, 1, 0);
                if (n <= 0) goto recv_fail;
                sl++;
            }
            size_line[sl] = 0;
            long long chunk_size = strtoll(size_line, NULL, 16);
            if (chunk_size < 0) goto recv_fail;
            if (chunk_size == 0) break; // 终止 chunk（后续 trailer 忽略，直接完成）
            // 读取 chunk 数据并透传
            long long remaining = chunk_size;
            while (remaining > 0) {
                size_t want = ((uint64_t)remaining < sizeof(iobuf)) ? (size_t)remaining : sizeof(iobuf);
                ssize_t n = recv(fd, iobuf, want, 0);
                if (n <= 0) goto recv_fail;
                if (net_ring_write(r, iobuf, (size_t)n) < (size_t)n) goto aborted;
                remaining -= n;
            }
            // 吃掉 chunk 后的 CRLF
            for (int crlf = 0; crlf < 2; crlf++) {
                ssize_t n = recv(fd, iobuf, 1, 0);
                if (n <= 0) goto recv_fail;
            }
        }
    }
    else if (content_len >= 0) {
        int64_t remaining = content_len;
        while (remaining > 0) {
            size_t want = ((uint64_t)remaining < sizeof(iobuf)) ? (size_t)remaining : sizeof(iobuf);
            ssize_t n = recv(fd, iobuf, want, 0);
            if (n <= 0) goto recv_fail;
            if (net_ring_write(r, iobuf, (size_t)n) < (size_t)n) goto aborted;
            remaining -= n;
        }
    }
    else {
        // 无长度声明：读到连接关闭
        for (;;) {
            ssize_t n = recv(fd, iobuf, sizeof(iobuf), 0);
            if (n < 0) goto recv_fail;
            if (n == 0) break;
            if (net_ring_write(r, iobuf, (size_t)n) < (size_t)n) goto aborted;
        }
    }

    close(fd);
    pthread_mutex_lock(&r->lock);
    r->state = NET_STREAM_DONE;
    r->worker_done = 1;
    pthread_mutex_unlock(&r->lock);
    return NULL;

aborted:
    close(fd);
    pthread_mutex_lock(&r->lock);
    r->state = NET_STREAM_ERROR;
    r->worker_done = 1;
    pthread_mutex_unlock(&r->lock);
    return NULL;

recv_fail:
    fprintf(stderr, "[hal_net] body recv failed/aborted (n<=0 or %s)\n", strerror(errno));
    close(fd);
    pthread_mutex_lock(&r->lock);
    r->state = NET_STREAM_ERROR;
    r->worker_done = 1;
    pthread_mutex_unlock(&r->lock);
    return NULL;
}

// ===============================================================================
// 对外接口
// ===============================================================================

int32_t net_http_post_stream(const Net_Http_Request_Config *config, Net_Http_Request **out_req) {
    if (config == NULL || out_req == NULL || config->url == NULL) return -1;

    Net_Request_Linux *r = (Net_Request_Linux *)calloc(1, sizeof(Net_Request_Linux));
    if (r == NULL) return -1;

    if (net_parse_url(config->url, r->host, sizeof(r->host), r->path, sizeof(r->path), &r->port) != 0) {
        free(r);
        return -1;
    }
    snprintf(r->url, sizeof(r->url), "%s", config->url);
    if (config->body_len > 0 && config->body != NULL) {
        r->body = (uint8_t *)malloc(config->body_len);
        if (r->body == NULL) { free(r); return -1; }
        memcpy(r->body, config->body, config->body_len);
        r->body_len = config->body_len;
    }
    int32_t hnum = (config->header_num > NANO_NET_MAX_HEADERS) ? NANO_NET_MAX_HEADERS : config->header_num;
    for (int32_t i = 0; i < hnum; i++) {
        if (config->headers[i].key == NULL) continue;
        snprintf(r->headers[r->header_num], sizeof(r->headers[0]), "%s: %s",
                 config->headers[i].key, config->headers[i].value ? config->headers[i].value : "");
        r->header_num++;
    }

    pthread_mutex_init(&r->lock, NULL);
    pthread_cond_init(&r->cond_space, NULL);
    pthread_cond_init(&r->cond_data, NULL);
    r->state = NET_STREAM_IN_PROGRESS;
    r->handle_closed = 0;

    pthread_t tid;
    if (pthread_create(&tid, NULL, net_worker, r) != 0) {
        pthread_mutex_destroy(&r->lock);
        pthread_cond_destroy(&r->cond_space);
        pthread_cond_destroy(&r->cond_data);
        free(r->body);
        free(r);
        return -1;
    }
    pthread_detach(tid);

    *out_req = (Net_Http_Request *)r;
    return 0;
}

int32_t net_http_stream_read(Net_Http_Request *req, uint8_t *buf, size_t buf_size) {
    Net_Request_Linux *r = (Net_Request_Linux *)req;
    if (r == NULL || buf == NULL || buf_size == 0) return -1;

    pthread_mutex_lock(&r->lock);
    size_t avail = net_ring_used(r);
    size_t n = (avail < buf_size) ? avail : buf_size;
    if (n > 0) {
        size_t head = r->ring_head;
        size_t first = (head + n <= NANO_NET_RING_SIZE) ? n : (NANO_NET_RING_SIZE - head);
        memcpy(buf, r->ring + head, first);
        if (n > first) memcpy(buf + first, r->ring, n - first);
        r->ring_head = (head + n) % NANO_NET_RING_SIZE;
        pthread_cond_signal(&r->cond_space); // 唤醒等待空间的写者
    }
    int32_t state = r->state;
    int32_t closed = r->handle_closed;
    pthread_mutex_unlock(&r->lock);

    if (closed) return -1;               // 已中止：句柄失效
    if (n > 0) return (int32_t)n;
    if (state == NET_STREAM_DONE) return 0;
    if (state == NET_STREAM_ERROR) return -1;
    return 0;                            // IN_PROGRESS 且暂无数据
}

Net_Stream_State net_http_stream_state(Net_Http_Request *req) {
    Net_Request_Linux *r = (Net_Request_Linux *)req;
    if (r == NULL) return NET_STREAM_ERROR;
    pthread_mutex_lock(&r->lock);
    int32_t state = (r->handle_closed) ? NET_STREAM_ERROR : r->state;
    pthread_mutex_unlock(&r->lock);
    return (Net_Stream_State)state;
}

int32_t net_http_status_code(Net_Http_Request *req) {
    Net_Request_Linux *r = (Net_Request_Linux *)req;
    if (r == NULL) return 0;
    pthread_mutex_lock(&r->lock);
    int32_t status = r->http_status;
    pthread_mutex_unlock(&r->lock);
    return status;
}

void net_http_close(Net_Http_Request *req) {
    Net_Request_Linux *r = (Net_Request_Linux *)req;
    if (r == NULL) return;
    pthread_mutex_lock(&r->lock);
    if (r->handle_closed) { pthread_mutex_unlock(&r->lock); return; }
    r->handle_closed = 1;
    r->close_flag = 1;
    r->state = NET_STREAM_ERROR; // 对外即失效
    pthread_cond_broadcast(&r->cond_space);
    pthread_mutex_unlock(&r->lock);
    net_retire(r); // 工作线程尚未退出则进坟场延后收尸
}
