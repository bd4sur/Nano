#!/usr/bin/env python3
"""极简 HTTPS 静态文件服务器（自签名证书；供 Nano-WASM-GUI 等静态产物本地托管）。

浏览器 DeviceMotion/DeviceOrientation 传感器 API 要求安全上下文（HTTPS/localhost），
故 WASM 页面若想在手机上用姿态功能，需经 HTTPS 托管。

远端模型同源反代（2026-10-14，配合鹦鹉笼“远端模型”选项）：
  浏览器同源策略下，页面无法直接跨源 fetch 远端模型 API（该 API 不支持 CORS
  预检且不返回 Access-Control-Allow-Origin）。本服务把同源路径 /v1/* 反向代理到
  上游 API（环境变量 NANO_API_PROXY 指定上游地址，不写入本仓库），流式转发
  （SSE 逐块透传），WASM 侧 hal_net 以同源相对路径 /v1 访问（见 ui_llm.c）。

用法：
  python3 https_static_server.py [端口=9002] [目录=.]
证书：默认使用 infer/bin/web_cert.pem + web_key.pem（tools/gen_web_cert.sh 生成，
SAN 含本机全部 IP）；环境变量 NANO_TLS_CERT / NANO_TLS_KEY 可覆盖。
"""
import http.client
import http.server
import os
import ssl
import sys
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
INFER_DIR = os.path.dirname(HERE)  # tools/ 的上一级（infer/）

port = int(sys.argv[1]) if len(sys.argv) > 1 else 9002
web_dir = sys.argv[2] if len(sys.argv) > 2 else "."

cert = os.environ.get("NANO_TLS_CERT", os.path.join(INFER_DIR, "bin", "web_cert.pem"))
key = os.environ.get("NANO_TLS_KEY", os.path.join(INFER_DIR, "bin", "web_key.pem"))

# 远端模型 API 反代（鹦鹉笼“远端模型”选项；同源路径 /v1/* → 上游）。
# 上游经环境变量 NANO_API_PROXY 指定（如 http://<内网API主机>:<端口>），不设默认值
# （源码不入库任何内网地址）；未设置时不启用反代（/v1/* 返回 503 提示）。
API_PROXY_UPSTREAM = os.environ.get("NANO_API_PROXY", "").rstrip("/")
API_PROXY_PREFIX = "/v1/"

# 反代时透传的请求头（其余请求头不属于业务语义，丢弃以免混淆上游）
API_PROXY_PASS_HEADERS = ("content-type", "accept", "authorization", "user-agent")
# 反代响应中不透传的头（逐跳/长度相关，由本服务重新决定）
API_PROXY_DROP_HEADERS = ("transfer-encoding", "content-length", "connection", "keep-alive", "server", "date")


class Handler(http.server.SimpleHTTPRequestHandler):
    # .wasm 必须带 application/wasm（WebAssembly.instantiateStreaming 的 MIME 要求）
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".data": "application/octet-stream",
    }

    def end_headers(self):
        # 禁用缓存：开发/迭代期浏览器（尤其手机 Chrome）会缓存旧版 wasm/js 导致改动不生效
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=web_dir, **kwargs)

    def log_message(self, fmt, *args):  # 精简日志
        sys.stderr.write("[https-static] %s - %s\n" % (self.address_string(), fmt % args))

    # ---------------- 远端模型 API 同源反代 ----------------
    def _is_api_proxy(self):
        return self.path.startswith(API_PROXY_PREFIX)

    def _api_proxy_request(self, method):
        if not API_PROXY_UPSTREAM:
            self.send_error(503, "API proxy not configured (set NANO_API_PROXY)")
            return
        parts = urlsplit(self.path)
        upstream = urlsplit(API_PROXY_UPSTREAM)
        host = upstream.hostname or "127.0.0.1"
        port_up = upstream.port or 80
        # 同源前缀与上游路径前缀一致（/v1/* 原样转发；上游 API base 即含 /v1）
        target_path = parts.path
        if parts.query:
            target_path += "?" + parts.query

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else None

        try:
            conn = http.client.HTTPConnection(host, port_up, timeout=300)
            conn.putrequest(method, target_path, skip_host=True, skip_accept_encoding=True)
            conn.putheader("Host", f"{host}:{port_up}")
            for hk, hv in self.headers.items():
                if hk.lower() in API_PROXY_PASS_HEADERS:
                    conn.putheader(hk, hv)
            if body is not None:
                conn.putheader("Content-Length", str(len(body)))
            conn.endheaders()
            if body is not None:
                conn.send(body)
            resp = conn.getresponse()
        except OSError as e:
            self.send_error(502, f"API proxy upstream unreachable: {e}")
            return

        self.send_response(resp.status)
        for hk, hv in resp.getheaders():
            if hk.lower() not in API_PROXY_DROP_HEADERS:
                self.send_header(hk, hv)
        self.send_header("Connection", "close")
        self.end_headers()
        # 流式透传（SSE 逐块转发，不缓冲整个响应）
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端提前断开（页面刷新/中止推理），属正常
        finally:
            conn.close()

    def do_GET(self):
        if self._is_api_proxy():
            return self._api_proxy_request("GET")
        return super().do_GET()

    def do_POST(self):
        if self._is_api_proxy():
            return self._api_proxy_request("POST")
        self.send_error(404)


httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(certfile=cert, keyfile=key)
httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
print(f"[https-static] serving {os.path.abspath(web_dir)} on https://0.0.0.0:{port}/ (cert={cert})")
if API_PROXY_UPSTREAM:
    print(f"[https-static] API proxy: /v1/ -> {API_PROXY_UPSTREAM}")
else:
    print("[https-static] API proxy disabled (set NANO_API_PROXY to enable /v1/ forwarding)")
httpd.serve_forever()
