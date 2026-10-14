#!/usr/bin/env python3
"""极简 HTTPS 静态文件服务器（自签名证书；供 Nano-WASM-GUI 等静态产物本地托管）。

浏览器 DeviceMotion/DeviceOrientation 传感器 API 要求安全上下文（HTTPS/localhost），
故 WASM 页面若想在手机上用姿态功能，需经 HTTPS 托管。

用法：
  python3 https_static_server.py [端口=9002] [目录=.]
证书：默认使用 infer/bin/web_cert.pem + web_key.pem（tools/gen_web_cert.sh 生成，
SAN 含本机全部 IP）；环境变量 NANO_TLS_CERT / NANO_TLS_KEY 可覆盖。
"""
import http.server
import os
import ssl
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INFER_DIR = os.path.dirname(HERE)  # tools/ 的上一级（infer/）

port = int(sys.argv[1]) if len(sys.argv) > 1 else 9002
web_dir = sys.argv[2] if len(sys.argv) > 2 else "."

cert = os.environ.get("NANO_TLS_CERT", os.path.join(INFER_DIR, "bin", "web_cert.pem"))
key = os.environ.get("NANO_TLS_KEY", os.path.join(INFER_DIR, "bin", "web_key.pem"))


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


httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(certfile=cert, keyfile=key)
httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
print(f"[https-static] serving {os.path.abspath(web_dir)} on https://0.0.0.0:{port}/ (cert={cert})")
httpd.serve_forever()
