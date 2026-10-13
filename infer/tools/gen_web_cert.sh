#!/bin/sh
# gen_web_cert.sh —— 为 nano_web 生成自签名 TLS 证书（HTTPS/WSS）
#
# 用途：浏览器 DeviceMotion/DeviceOrientation 等传感器 API 仅在安全上下文
# （HTTPS 或 localhost）可用；局域网 HTTP 页面无法获取设备姿态。
# 本脚本生成自签名证书（SAN 自动包含 localhost 与本机全部 IP），输出至 bin/：
#   bin/web_cert.pem / bin/web_key.pem（bin/ 已被 .gitignore，私钥不进仓库）
#
# 使用：sh tools/gen_web_cert.sh   （在 infer/ 目录下执行；IP 变化后重新执行）
# 浏览器首次访问 https://<主机IP>:9002/ 会有证书警告，点击「继续前往」即可
# （同源的 WSS 连接共用该证书例外）。如需彻底消除警告，可将 web_cert.pem
# 安装为手机/电脑的受信根证书。

set -e

OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/bin"
mkdir -p "$OUT_DIR"
CERT="$OUT_DIR/web_cert.pem"
KEY="$OUT_DIR/web_key.pem"

# SAN：localhost + 本机全部 IPv4 地址
SAN="DNS:localhost,IP:127.0.0.1"
for ip in $(hostname -I 2>/dev/null); do
    case "$ip" in
        *.*) SAN="$SAN,IP:$ip" ;;
    esac
done

openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$KEY" -out "$CERT" -days 3650 \
    -subj "/CN=nano-web" \
    -addext "subjectAltName=$SAN" 2>/dev/null

chmod 600 "$KEY"
echo "generated: $CERT"
echo "           $KEY"
echo "SAN: $SAN"
