#ifndef __NANO_HAL_NET_H__
#define __NANO_HAL_NET_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===============================================================================
// hal_net：网络通信抽象层（硬件无关接口）
//   为"远端模型"等业务提供对云端 HTTP API 的访问能力。与既有 HAL 同构：
//   接口头 hal_net.h + 各平台实现 hal_net_<平台>.c：
//     - hal_net_linux.c     POSIX/Linux（tty、web 等本机目标；后台线程 + 环形缓冲）
//     - hal_net_wasm.c      浏览器内 Emscripten（fetch + ReadableStream）
//     - hal_net_stub.c      其余平台打桩（恒失败，调用方按"网络不可用"降级）
//
// 接口为非阻塞/轮询式，与整机主循环的协作式调度模型契合（推理、渲染与网络
// 共享一个事件循环，任何一步都不得长时间阻塞）：
//   1. net_http_post_stream() 发起一个流式 HTTP 请求后【立即返回】；
//   2. 业务在主循环中反复调用 net_http_stream_read() 拉取响应体数据块：
//        返回值 >0          本次读到的字节数（可继续读）
//        返回值 0           暂无新数据（配合状态判断，见下）
//        返回值 -1          流错误（HTTP 层失败/连接中断/请求已中止）
//      流结束的判定：net_http_stream_state() == NET_STREAM_DONE（此后 read 恒 0）；
//   3. net_http_close() 随时终止并释放请求（中止未完成的传输；幂等）。
//
// 能力边界（保持接口最小内聚，协议编组不属于本层）：
//   - 仅支持明文 HTTP/1.1（远端模型 API 为 http:// 内网服务）；
//   - 响应体按原始字节流透传（含 Content-Length / chunked / 连接关闭三种分帧），
//     SSE 等应用层协议由调用方解析；
//   - 单请求单连接，不做连接复用/重定向/代理。
// ===============================================================================

// 流状态
typedef enum {
    NET_STREAM_IDLE = 0,        // 刚创建，尚未开始传输
    NET_STREAM_IN_PROGRESS = 1, // 传输中
    NET_STREAM_DONE = 2,        // 正常完成（响应体已全部读出）
    NET_STREAM_ERROR = 3        // 失败（连接失败/中断/请求被中止）
} Net_Stream_State;

// 不透明请求句柄（业务不得解引用；经下方函数操作）
typedef void Net_Http_Request;

// 请求头键值对
typedef struct {
    const char *key;
    const char *value;
} Net_Http_Header;

#define NANO_NET_MAX_HEADERS (8)

// 请求配置（net_http_post_stream 调用后本结构即可释放；实现内部会深拷贝所需内容）
typedef struct {
    const char *url;      // 完整 URL，当前支持 http://host[:port]/path
    const uint8_t *body;  // 请求体（可为 NULL，此时 body_len 须为 0）
    size_t body_len;
    Net_Http_Header headers[NANO_NET_MAX_HEADERS]; // 附加请求头（Host/Content-Length 等由实现自动补）
    int32_t header_num;
} Net_Http_Request_Config;

// 发起流式 HTTP 请求（当前实现固定为 POST；非阻塞，立即返回）。
// 成功返回 0 且 *out_req 为有效句柄；失败（参数非法/资源不足/平台不支持）返回 -1。
int32_t net_http_post_stream(const Net_Http_Request_Config *config, Net_Http_Request **out_req);

// 非阻塞读取响应体数据块（至多 buf_size 字节写入 buf）。
// 返回值：>0 实际字节数；0 暂无新数据或流已正常结束（以状态为准）；-1 错误。
int32_t net_http_stream_read(Net_Http_Request *req, uint8_t *buf, size_t buf_size);

// 查询流状态（req 为 NULL 时返回 NET_STREAM_ERROR）
Net_Stream_State net_http_stream_state(Net_Http_Request *req);

// HTTP 状态码（收到响应头后有效；尚未收到/未知时为 0）
int32_t net_http_status_code(Net_Http_Request *req);

// 终止并释放请求。任何状态（含进行中/已完成/已失败）均可调用，幂等；
// 调用后句柄立即失效，不得再使用。进行中的传输会被中止（尽力而为）。
void net_http_close(Net_Http_Request *req);

#ifdef __cplusplus
}
#endif

#endif
