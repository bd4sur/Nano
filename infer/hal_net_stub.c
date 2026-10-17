// hal_net_stub.c —— hal_net 网络通信 HAL 的打桩实现（无网络能力平台）
//
// 适用：暂无网络通路/暂未实现网络 HAL 的编译目标（Nano-Pod 系列嵌入式平台等）。
// 语义：所有请求一律失败（net_http_post_stream 返回 -1；read 返回 -1；
//       state 恒为 NET_STREAM_ERROR）。业务调用方（远端模型适配层等）按
//       "网络不可用"路径降级，显示错误提示，不影响其余功能。

#include "hal_net.h"

#include <stdio.h>

int32_t net_http_post_stream(const Net_Http_Request_Config *config, Net_Http_Request **out_req) {
    (void)config;
    if (out_req != NULL) *out_req = NULL;
    fprintf(stderr, "[hal_net] stub: network not supported on this platform\n");
    return -1;
}

int32_t net_http_stream_read(Net_Http_Request *req, uint8_t *buf, size_t buf_size) {
    (void)req; (void)buf; (void)buf_size;
    return -1;
}

Net_Stream_State net_http_stream_state(Net_Http_Request *req) {
    (void)req;
    return NET_STREAM_ERROR;
}

int32_t net_http_status_code(Net_Http_Request *req) {
    (void)req;
    return 0;
}

void net_http_close(Net_Http_Request *req) {
    (void)req;
}
