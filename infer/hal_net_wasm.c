// hal_net_wasm.c —— hal_net 网络通信 HAL 的浏览器内 Emscripten 实现
//
// 服务对象：Nano-WASM-GUI（NANO_WASM_GUI，单线程构建）。浏览器主线程上 JS 与
//   WASM 交错执行：JS 侧的 fetch 流式读取循环（异步）与 WASM 主循环的轮询
//   （rAF 帧回调内同步）天然分时——JS 循环每次 await 让出事件循环时，浏览器
//   处理 WASM 的下一帧；WASM 帧回调内 net_http_stream_read 同步取走已到达的
//   数据块。数据通路：JS 把每个 chunk 缓存进本请求的队列，WASM 读取时从队列
//   头拷贝入线性内存（chunk 大于缓冲时拆分，余量留在队首）。
//
// URL 说明：浏览器存在同源策略，跨源 fetch 需目标服务器返回 CORS 头。
//   远端模型 API（http://um870p...:11050）不支持 CORS 预检（OPTIONS 501），
//   故 WASM 目标的业务经测试服务的同源反向代理访问（详见 ui_llm.c 远端模型
//   适配层与 tools/https_static_server.py 的 /v1 代理）——本 HAL 只负责按
//   调用方给定的 URL 发起 fetch，不感知代理。
//
// 中止：close 经 AbortController 取消 fetch（连接立即释放）。

#include "hal_net.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <emscripten.h>

// C 侧句柄：仅为 JS 侧请求对象的数字 id（JS 对象存于 Module._nano_net.reqs）
typedef struct {
    int32_t js_id;
} Net_Request_Wasm;

// JS 桥（EM_JS 编译进 glue；Module.HEAPU8 等经 Module 对象访问，
// 与 hal_display_wasm.c 的既有用法一致）

// 发起流式 POST：headers 为已序列化的 "Key: Value\nKey2: Value2" 文本
EM_JS(void, wasm_net_post_js,
      (int32_t js_id, const uint8_t *url, uint32_t url_len,
       const uint8_t *hdrs, uint32_t hdrs_len,
       const uint8_t *body, uint32_t body_len),
{
  if (!Module["_nano_net"]) {
    Module["_nano_net"] = { reqs: {}, next_id: 1 };
  }
  const net = Module["_nano_net"];
  const dec = new TextDecoder();
  const urlStr = dec.decode(Module.HEAPU8.subarray(url, url + url_len));
  const hdrStr = dec.decode(Module.HEAPU8.subarray(hdrs, hdrs + hdrs_len));
  const bodyBytes = Module.HEAPU8.subarray(body, body + body_len).slice(); // 拷出，防线性内存增长失效
  const headers = {};
  hdrStr.split("\n").forEach((line) => {
    const idx = line.indexOf(":");
    if (idx > 0) headers[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
  });
  const ctl = new AbortController();
  const rec = { queue: [], state: 1 /* IN_PROGRESS */, status: 0, ctl: ctl };
  net.reqs[js_id] = rec;
  (async () => {
    try {
      const resp = await fetch(urlStr, {
        method: "POST",
        headers: headers,
        body: bodyBytes,
        signal: ctl.signal,
      });
      rec.status = resp.status;
      if (!resp.ok) {
        rec.state = 3; // ERROR
        return;
      }
      if (!resp.body) {
        // 无流的响应（异常路径）：整体读入
        const buf = new Uint8Array(await resp.arrayBuffer());
        rec.queue.push(buf);
        rec.state = 2; // DONE
        return;
      }
      const reader = resp.body.getReader();
      for (;;) {
        const r = await reader.read();
        if (r.done) { rec.state = 2; break; } // DONE
        rec.queue.push(r.value);
      }
    } catch (e) {
      rec.state = 3; // ERROR（网络失败/被中止）
    }
  })();
});

// 非阻塞读取：从队首拷贝至 buf（chunk 大于缓冲时拆分，余量留在队首）。
// 返回拷贝字节数；0=暂无数据；-1=错误。状态经 wasm_net_state_js 单独查询。
EM_JS(int32_t, wasm_net_read_js, (int32_t js_id, uint8_t *buf, uint32_t buf_size), {
  const net = Module["_nano_net"];
  if (!net) return -1;
  const rec = net.reqs[js_id];
  if (!rec) return -1;
  if (rec.state === 3) return -1;
  if (rec.queue.length === 0) return 0;
  let chunk = rec.queue[0];
  const n = Math.min(chunk.length, buf_size);
  Module.HEAPU8.set(chunk.subarray(0, n), buf);
  if (n < chunk.length) {
    rec.queue[0] = chunk.subarray(n); // 余量留队首
  }
  else {
    rec.queue.shift();
  }
  return n;
});

EM_JS(int32_t, wasm_net_state_js, (int32_t js_id), {
  const net = Module["_nano_net"];
  if (!net) return 3;
  const rec = net.reqs[js_id];
  if (!rec) return 3;
  return rec.state;
});

EM_JS(int32_t, wasm_net_status_js, (int32_t js_id), {
  const net = Module["_nano_net"];
  if (!net) return 0;
  const rec = net.reqs[js_id];
  if (!rec) return 0;
  return rec.status | 0;
});

EM_JS(void, wasm_net_close_js, (int32_t js_id), {
  const net = Module["_nano_net"];
  if (!net) return;
  const rec = net.reqs[js_id];
  if (!rec) return;
  try { rec.ctl.abort(); } catch (e) {}
  delete net.reqs[js_id];
});

// JS 侧请求 id（单调递增；由 JS 维护计数器）
EM_JS(int32_t, wasm_net_next_id_js, (void), {
  if (!Module["_nano_net"]) Module["_nano_net"] = { reqs: {}, next_id: 1 };
  return Module["_nano_net"].next_id++;
});

// ---------------- 对外接口 ----------------

int32_t net_http_post_stream(const Net_Http_Request_Config *config, Net_Http_Request **out_req) {
    if (config == NULL || out_req == NULL || config->url == NULL) return -1;

    // 序列化请求头为 "Key: Value\n" 文本
    char hdrs[1024];
    size_t off = 0;
    int32_t hnum = config->header_num;
    if (hnum > NANO_NET_MAX_HEADERS) hnum = NANO_NET_MAX_HEADERS;
    for (int32_t i = 0; i < hnum; i++) {
        if (config->headers[i].key == NULL) continue;
        int w = snprintf(hdrs + off, sizeof(hdrs) - off, "%s: %s\n",
                         config->headers[i].key, config->headers[i].value ? config->headers[i].value : "");
        if (w < 0 || (size_t)w >= sizeof(hdrs) - off) break;
        off += (size_t)w;
    }

    Net_Request_Wasm *h = (Net_Request_Wasm *)calloc(1, sizeof(Net_Request_Wasm));
    if (h == NULL) return -1;

    h->js_id = wasm_net_next_id_js();

    wasm_net_post_js(h->js_id,
                     (const uint8_t *)config->url, (uint32_t)strlen(config->url),
                     (const uint8_t *)hdrs, (uint32_t)off,
                     config->body, (uint32_t)config->body_len);

    *out_req = (Net_Http_Request *)h;
    return 0;
}

int32_t net_http_stream_read(Net_Http_Request *req, uint8_t *buf, size_t buf_size) {
    Net_Request_Wasm *h = (Net_Request_Wasm *)req;
    if (h == NULL || buf == NULL || buf_size == 0) return -1;
    return wasm_net_read_js(h->js_id, buf, (uint32_t)buf_size);
}

Net_Stream_State net_http_stream_state(Net_Http_Request *req) {
    Net_Request_Wasm *h = (Net_Request_Wasm *)req;
    if (h == NULL) return NET_STREAM_ERROR;
    return (Net_Stream_State)wasm_net_state_js(h->js_id);
}

int32_t net_http_status_code(Net_Http_Request *req) {
    Net_Request_Wasm *h = (Net_Request_Wasm *)req;
    if (h == NULL) return 0;
    return wasm_net_status_js(h->js_id);
}

void net_http_close(Net_Http_Request *req) {
    Net_Request_Wasm *h = (Net_Request_Wasm *)req;
    if (h == NULL) return;
    wasm_net_close_js(h->js_id);
    free(h);
}
