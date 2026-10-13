#ifndef __NANO_WS_SERVER_H__
#define __NANO_WS_SERVER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

// ===============================================================================
// Nano-Web 远程桌面服务器（极简 HTTP + WebSocket，RFC 6455 子集，无第三方依赖）
//
// C/S 分离架构（编译目标 nano_web / 平台宏 NANO_WEB）：
//   服务端（本模块 + 各 HAL）：运行完整 GUI 渲染与交互逻辑，
//     - 显示 HAL（hal_display_web_linux.c）把 gfx_refresh 的脏矩形经本模块广播；
//     - 触屏 HAL（hal_touch_web_linux.c）把本模块收到的触屏事件喂给上层轮询。
//   客户端（web_remote_client.html，薄客户端，类远程桌面）：
//     只负责把帧缓冲画到 <canvas>、把触屏/鼠标事件发回来。
//
// 协议（WebSocket 二进制消息，多字节整数一律小端）：
//   S→C  [0x00][w u16][h u16]          初始化：帧缓冲尺寸（连接建立后首条消息）
//   S→C  [0x03][seq u16][x u16][y u16][w u16][h u16][RGBA8888 像素 w*h*4]  帧缓冲脏矩形
//        （seq 为发送序号，客户端可统计跳帧；RGBA 由服务端展开，客户端一次原生拷贝上屏。
//         推送按固定节奏（默认 60fps，NANO_WEB_MAX_FPS 可调）合并高频刷新，
//         避免浏览器事件循环积压；v1 的 0x01/RGB888 消息已退役）
//   C→S  [0x10][action u8][x i16][y i16]  触屏事件：action 0=松开 1=按下 2=移动
//   C→S  [0x20][flags u8][ax f32][ay f32][az f32][yaw f32][orient i16]  姿态事件：
//        flags bit0=加速度有效 bit1=航向有效；ax/ay/az 为浏览器设备坐标系重力向量（m/s²，
//        DeviceMotionEvent.accelerationIncludingGravity）；yaw 为浏览器罗盘航向 alpha（度）；
//        orient 为屏幕方向角 screen.orientation.angle（0/90/180/270）。角度解算在 IMU HAL。
// ===============================================================================

// 启动服务器（监听 0.0.0.0:port；fb_width/fb_height 为帧缓冲像素尺寸）。
// 成功返回 0；失败返回负值（端口占用等）。幂等：重复调用先停后启。
int32_t nano_ws_server_start(uint16_t port, uint32_t fb_width, uint32_t fb_height);

// 停止服务器（关闭监听与全部客户端连接；进程退出前的尽力而为清理）
void nano_ws_server_stop(void);

// 当前已建立 WebSocket 连接的客户端数（诊断用）
int32_t nano_ws_server_client_count(void);

// 推送帧缓冲脏矩形（RGB888，行步进 fb_width）。由显示 HAL 在 gfx_refresh 路径调用。
// 内部并入全局镜像并向各客户端累积脏区；实际发送由各客户端独立的发送线程完成——
// 网络慢/客户端卡顿时天然按接收方带宽合并降频，不会反压渲染线程。
void nano_ws_server_push_frame(const uint8_t *rgb888, uint32_t fb_width, uint32_t fb_height,
                               uint32_t x, uint32_t y, uint32_t w, uint32_t h);

// 触屏事件入口：由触屏 HAL（hal_touch_web_linux.c）实现，本模块的客户端接收线程调用。
// action：0=松开 1=按下 2=移动；x/y 为帧缓冲像素坐标。
void web_remote_on_touch(int32_t action, int32_t x, int32_t y);

// 姿态事件入口：由 IMU HAL（hal_imu_web_linux.c）实现，本模块的客户端接收线程调用。
// acc：设备坐标系重力向量 [3]（m/s²），可为 NULL（flags bit0=0 时）；yaw_deg：罗盘航向（度）；
// orient_deg：屏幕方向角；flags：bit0=加速度有效 bit1=航向有效。
void web_remote_on_orientation(const float *acc, float yaw_deg, int32_t orient_deg, uint8_t flags);

#ifdef __cplusplus
}
#endif

#endif
