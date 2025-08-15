#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <alsa/asoundlib.h>
#include <libwebsockets.h>
#include <json-c/json.h>

#define SAMPLE_RATE 16000
#define CHANNELS 2
#define FRAME_SIZE 960  // 960字节 = 480 samples * 2 bytes/sample * 1 channel (单声道)
#define AUDIO_DEVICE "default"

// 全局变量
static struct lws *websocket_client = NULL;
static struct lws_context *context = NULL;
static int websocket_connected = 0;
static int websocket_ready = 0;
static snd_pcm_t *capture_handle = NULL;
static pthread_t audio_thread;
static volatile int running = 1;
static volatile int recording = 0;
static char *websocket_url = "ws://127.0.0.1:10096";

// WebSocket回调函数
static int websocket_callback(struct lws *wsi, enum lws_callback_reasons reason,
                             void *user, void *in, size_t len) {
    switch (reason) {
        case LWS_CALLBACK_CLIENT_ESTABLISHED:
            printf("WebSocket连接已建立\n");
            websocket_connected = 1;
            websocket_ready = 1;
            break;
            
        case LWS_CALLBACK_CLIENT_RECEIVE:
            if (in && len > 0) {
                char *response = malloc(len + 1);
                if (response) {
                    memcpy(response, in, len);
                    response[len] = '\0';
                    printf("ASR识别结果: %s\n", response);
                    free(response);
                }
            }
            break;
            
        case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
            printf("WebSocket连接错误: ");
            if (in) {
                printf("%s\n", (char *)in);
            } else {
                printf("未知错误\n");
            }
            websocket_connected = 0;
            websocket_ready = 0;
            running = 0;
            break;
            
        case LWS_CALLBACK_CLOSED:
            printf("WebSocket连接已关闭\n");
            websocket_connected = 0;
            websocket_ready = 0;
            break;
            
        case LWS_CALLBACK_CLIENT_WRITEABLE:
            // 连接建立后发送初始化JSON
            if (websocket_connected && websocket_ready) {
                struct json_object *json_obj = json_object_new_object();
                struct json_object *chunk_size = json_object_new_array();
                
                json_object_array_add(chunk_size, json_object_new_int(5));
                json_object_array_add(chunk_size, json_object_new_int(10));
                json_object_array_add(chunk_size, json_object_new_int(5));
                
                json_object_object_add(json_obj, "chunk_size", chunk_size);
                json_object_object_add(json_obj, "wav_name", json_object_new_string("h5"));
                json_object_object_add(json_obj, "is_speaking", json_object_new_boolean(1));
                json_object_object_add(json_obj, "chunk_interval", json_object_new_int(10));
                json_object_object_add(json_obj, "itn", json_object_new_boolean(0));
                json_object_object_add(json_obj, "mode", json_object_new_string("online"));
                json_object_object_add(json_obj, "hotwords", json_object_new_string("{\"hello world\": 40}"));
                
                const char *json_str = json_object_to_json_string(json_obj);
                int json_len = strlen(json_str);
                
                unsigned char *buf = malloc(LWS_SEND_BUFFER_PRE_PADDING + json_len + LWS_SEND_BUFFER_POST_PADDING);
                if (buf) {
                    memcpy(&buf[LWS_SEND_BUFFER_PRE_PADDING], json_str, json_len);
                    int result = lws_write(wsi, &buf[LWS_SEND_BUFFER_PRE_PADDING], json_len, LWS_WRITE_TEXT);
                    if (result < 0) {
                        printf("发送初始化JSON失败\n");
                    } else {
                        printf("已发送初始化JSON: %s\n", json_str);
                    }
                    free(buf);
                }
                
                json_object_put(json_obj);
                websocket_ready = 0; // 只发送一次
            }
            break;
            
        case LWS_CALLBACK_WSI_DESTROY:
            printf("WebSocket连接实例已销毁\n");
            websocket_connected = 0;
            websocket_ready = 0;
            break;
            
        default:
            break;
    }
    
    return 0;
}

// WebSocket协议定义
static struct lws_protocols protocols[] = {
    {
        "",
        websocket_callback,
        0,
        4096,
    },
    { NULL, NULL, 0, 0 }
};

// 初始化音频设备
int init_audio_capture() {
    snd_pcm_hw_params_t *hw_params;
    int err;
    
    printf("正在打开音频设备 '%s'...\n", AUDIO_DEVICE);
    
    // 打开音频设备
    if ((err = snd_pcm_open(&capture_handle, AUDIO_DEVICE, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf(stderr, "无法打开音频设备 '%s': %s\n", AUDIO_DEVICE, snd_strerror(err));
        return -1;
    }
    
    // 分配硬件参数结构体
    snd_pcm_hw_params_alloca(&hw_params);
    
    // 初始化硬件参数
    if ((err = snd_pcm_hw_params_any(capture_handle, hw_params)) < 0) {
        fprintf(stderr, "无法初始化硬件参数: %s\n", snd_strerror(err));
        return -1;
    }
    
    // 设置访问类型为交错模式
    if ((err = snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        fprintf(stderr, "无法设置访问类型: %s\n", snd_strerror(err));
        return -1;
    }
    
    // 设置采样格式为16位有符号整数小端序
    if ((err = snd_pcm_hw_params_set_format(capture_handle, hw_params, SND_PCM_FORMAT_S16_LE)) < 0) {
        fprintf(stderr, "无法设置采样格式: %s\n", snd_strerror(err));
        return -1;
    }
    
    // 设置采样率
    unsigned int rate = SAMPLE_RATE;
    int dir = 0;
    if ((err = snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &rate, &dir)) < 0) {
        fprintf(stderr, "无法设置采样率: %s\n", snd_strerror(err));
        return -1;
    }
    printf("实际设置的采样率: %u Hz\n", rate);
    
    // 设置声道数为1（单声道，根据JavaScript代码）
    if ((err = snd_pcm_hw_params_set_channels(capture_handle, hw_params, 1)) < 0) {
        fprintf(stderr, "无法设置声道数为1: %s\n", snd_strerror(err));
        // 尝试设置为2声道
        if ((err = snd_pcm_hw_params_set_channels(capture_handle, hw_params, 2)) < 0) {
            fprintf(stderr, "无法设置声道数为2: %s\n", snd_strerror(err));
            return -1;
        }
        printf("设置为双声道\n");
    } else {
        printf("设置为单声道\n");
    }
    
    // 设置周期大小
    snd_pcm_uframes_t period_size = FRAME_SIZE / 2; // 480 samples for 16-bit mono
    if ((err = snd_pcm_hw_params_set_period_size_near(capture_handle, hw_params, &period_size, &dir)) < 0) {
        fprintf(stderr, "无法设置周期大小: %s\n", snd_strerror(err));
        return -1;
    }
    
    // 应用硬件参数
    if ((err = snd_pcm_hw_params(capture_handle, hw_params)) < 0) {
        fprintf(stderr, "无法应用硬件参数: %s\n", snd_strerror(err));
        return -1;
    }
    
    // 准备音频设备
    if ((err = snd_pcm_prepare(capture_handle)) < 0) {
        fprintf(stderr, "无法准备音频设备: %s\n", snd_strerror(err));
        return -1;
    }
    
    printf("音频设备初始化成功\n");
    return 0;
}

// 音频采集线程
void* audio_capture_thread(void *arg) {
    short buffer[FRAME_SIZE / 2]; // 960字节 = 480个short值（单声道）
    int frames_per_period = FRAME_SIZE / 2; // 480个采样点
    
    printf("音频采集线程已启动\n");
    
    while (running) {
        if (!recording) {
            usleep(10000); // 10ms
            continue;
        }
        
        // 读取音频数据
        int frames = snd_pcm_readi(capture_handle, buffer, frames_per_period);
        
        if (frames < 0) {
            if (frames == -EPIPE) {
                printf("音频缓冲区欠载，正在恢复...\n");
                snd_pcm_recover(capture_handle, frames, 0);
            } else {
                fprintf(stderr, "音频读取错误: %s\n", snd_strerror(frames));
                break;
            }
            continue;
        }
        
        if (frames > 0 && websocket_connected) {
            // 发送音频数据到WebSocket服务器
            unsigned char *buf = malloc(LWS_SEND_BUFFER_PRE_PADDING + FRAME_SIZE + LWS_SEND_BUFFER_POST_PADDING);
            if (buf) {
                memcpy(&buf[LWS_SEND_BUFFER_PRE_PADDING], buffer, FRAME_SIZE);
                
                if (websocket_client) {
                    int result = lws_write(websocket_client, &buf[LWS_SEND_BUFFER_PRE_PADDING], FRAME_SIZE, LWS_WRITE_BINARY);
                    if (result < 0) {
                        printf("发送音频数据失败\n");
                    }
                }
                
                free(buf);
            }
        }
    }
    
    printf("音频采集线程已退出\n");
    return NULL;
}

// 初始化WebSocket连接
int init_websocket() {
    struct lws_context_creation_info info;
    memset(&info, 0, sizeof(info));
    
    info.port = CONTEXT_PORT_NO_LISTEN;
    info.protocols = protocols;
    info.gid = -1;
    info.uid = -1;
    info.options = 0;
    
    context = lws_create_context(&info);
    if (!context) {
        fprintf(stderr, "无法创建libwebsockets上下文\n");
        return -1;
    }
    
    // 解析URL
    char address[256] = "127.0.0.1";
    int port = 10096;
    char path[256] = "/";

    printf("正在连接到WebSocket服务器 %s:%d%s...\n", address, port, path);
    
    // 连接到WebSocket服务器
    struct lws_client_connect_info ccinfo;
    memset(&ccinfo, 0, sizeof(ccinfo));
    
    ccinfo.context = context;
    ccinfo.address = address;
    ccinfo.port = port;
    ccinfo.path = path;
    ccinfo.host = address;
    ccinfo.origin = address;
    ccinfo.protocol = "";
    
    websocket_client = lws_client_connect_via_info(&ccinfo);
    if (!websocket_client) {
        fprintf(stderr, "无法连接到WebSocket服务器 %s:%d\n", address, port);
        return -1;
    }
    
    return 0;
}

// 主循环
void main_loop() {
    printf("进入主循环...\n");
    while (running) {
        lws_service(context, 50); // 50ms超时
    }
}

// 开始录音
void start_recording() {
    recording = 1;
    printf("开始录音\n");
}

// 停止录音
void stop_recording() {
    recording = 0;
    printf("停止录音\n");
}

// 连接到ASR服务
int connect_asr() {
    if (init_websocket() < 0) {
        return -1;
    }
    
    // 等待连接建立
    int timeout = 50; // 5秒超时
    while (timeout > 0 && !websocket_connected) {
        lws_service(context, 100);
        usleep(100000); // 100ms
        timeout--;
    }
    
    if (!websocket_connected) {
        fprintf(stderr, "WebSocket连接超时\n");
        return -1;
    }
    
    return 0;
}

// 断开连接
void disconnect_asr() {
    if (websocket_client) {
        lws_close_reason(websocket_client, LWS_CLOSE_STATUS_NORMAL, NULL, 0);
    }
    printf("已断开ASR连接\n");
}

// 信号处理函数
void signal_handler(int sig) {
    printf("\n正在关闭程序...\n");
    running = 0;
    recording = 0;
}

// 清理资源
void cleanup() {
    printf("正在清理资源...\n");
    running = 0;
    recording = 0;
    
    if (audio_thread) {
        pthread_join(audio_thread, NULL);
    }
    
    if (capture_handle) {
        snd_pcm_close(capture_handle);
    }
    
    if (context) {
        lws_context_destroy(context);
    }
    
    printf("资源已清理\n");
}

int main(int argc, char *argv[]) {
    // 注册信号处理函数
    // signal(SIGINT, signal_handler);
    // signal(SIGTERM, signal_handler);
    
    printf("=== 树莓派实时语音转写客户端 ===\n");
    printf("按 Ctrl+C 退出程序\n\n");
    
    // 初始化音频设备
    if (init_audio_capture() < 0) {
        fprintf(stderr, "音频设备初始化失败\n");
        return -1;
    }
    
    // 启动音频采集线程
    if (pthread_create(&audio_thread, NULL, audio_capture_thread, NULL) != 0) {
        fprintf(stderr, "无法创建音频采集线程\n");
        cleanup();
        return -1;
    }
    
    // 连接到ASR服务
    printf("正在连接到ASR服务...\n");
    if (connect_asr() < 0) {
        fprintf(stderr, "ASR服务连接失败\n");
        cleanup();
        return -1;
    }
    
    // 开始录音
    start_recording();
    
    // 运行主循环
    main_loop();
    
    // 停止录音
    stop_recording();
    
    // 断开连接
    disconnect_asr();
    
    // 清理资源
    cleanup();
    
    return 0;
}