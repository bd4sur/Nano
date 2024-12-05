import os
import time
import gc
import base64
from multiprocessing import Process
from threading import Thread

import ssl
from http.server import socketserver, SimpleHTTPRequestHandler


USE_SSL = True

SERVER_IP = '0.0.0.0'
HTTPS_PORT = 8443 if USE_SSL else 8088
API_PORT = 5000

SSL_CERT_PATH = "/home/bd4sur/bd4sur.crt"
SSL_PRIVATE_KEY_PATH = "/home/bd4sur/key_unencrypted.pem"


def start_https_server():
    httpd = socketserver.TCPServer((SERVER_IP, HTTPS_PORT), SimpleHTTPRequestHandler)
    if USE_SSL:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(SSL_CERT_PATH, SSL_PRIVATE_KEY_PATH)
        httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)
    print(f"Started HTTPS Server {SERVER_IP}:{HTTPS_PORT}")
    httpd.serve_forever()


if __name__ == '__main__':
    # HTTPS Server
    print("HTTPS")
    https_server_process = Process(target=start_https_server)
    https_server_process.daemon = True
    https_server_process.start()
    https_server_process.join()