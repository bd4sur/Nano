#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <stdint.h>
#include <signal.h>

#include "imu.h"

// ================= 配置区域 =================
// 串口设备路径，USB 转 TTL 通常为 /dev/ttyUSB0，GPIO 直连通常为 /dev/ttyAMA0
#define SERIAL_DEVICE "/dev/ttyAS4" 
// 波特率，根据 PCB 焊接点选择，默认为 115200
#define BAUDRATE 115200
// ===========================================

int fd = -1; // 串口文件描述符

// 角度数据结构
typedef struct {
    float yaw;   // 航向
    float pitch; // 俯仰
    float roll;  // 横滚
} SensorData;

// 终止程序处理
void signal_handler(int sig) {
    if (fd != -1) close(fd);
    printf("\n程序已退出。\n");
    exit(0);
}

// 初始化串口
int serial_init() {
    struct termios options;
    
    // 打开串口，O_RDWR 读写，O_NOCTTY 不作为控制终端，O_NDELAY 非阻塞 (稍后我们会配置阻塞)
    fd = open(SERIAL_DEVICE, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        perror("无法打开串口");
        return -1;
    }

    // 获取当前串口参数
    if (tcgetattr(fd, &options) != 0) {
        perror("获取串口属性失败");
        return -1;
    }

    // 设置波特率
    speed_t speed;
    switch(BAUDRATE) {
        case 9600: speed = B9600; break;
        case 19200: speed = B19200; break;
        case 38400: speed = B38400; break;
        case 57600: speed = B57600; break;
        case 115200: speed = B115200; break;
        default: speed = B115200; break;
    }
    cfsetispeed(&options, speed);
    cfsetospeed(&options, speed);

    // 设置数据位、校验位、停止位 (8N1)
    options.c_cflag |= (CLOCAL | CREAD); // 启用接收器，本地连接
    options.c_cflag &= ~PARENB;          // 无校验
    options.c_cflag &= ~CSTOPB;          // 1 位停止位
    options.c_cflag &= ~CSIZE;           // 清除数据位掩码
    options.c_cflag |= CS8;              // 8 位数据位
    options.c_cflag &= ~CRTSCTS;         // 无硬件流控制

    // 设置原始输入输出
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;

    // 设置读取行为：VMIN=1, VTIME=0 表示至少读取 1 个字节，无超时
    options.c_cc[VMIN] = 1;
    options.c_cc[VTIME] = 0;

    // 清除串口缓冲区中的旧数据
    tcflush(fd, TCIFLUSH);

    // 应用设置
    if (tcsetattr(fd, TCSANOW, &options) != 0) {
        perror("设置串口属性失败");
        return -1;
    }

    printf("串口 %s 初始化成功 (波特率：%d)\n", SERIAL_DEVICE, BAUDRATE);
    return 0;
}

// 发送命令
int send_command(uint8_t cmd1, uint8_t cmd2) {
    uint8_t buf[2] = {cmd1, cmd2};
    if (write(fd, buf, 2) != 2) {
        perror("发送命令失败");
        return -1;
    }
    // 稍微延时，等待模块处理
    usleep(50000);
    return 0;
}

// 读取一帧数据 (8 字节)，带帧头帧尾校验
int read_frame(uint8_t *buffer) {
    uint8_t byte;
    int state = 0; // 状态机：0=找帧头，1-6=收数据，7=验帧尾
    int count = 0;
    
    // 清空输入缓冲区，避免读取到旧数据
    tcflush(fd, TCIFLUSH);

    while (1) {
        if (read(fd, &byte, 1) <= 0) {
            // 读取失败或无数据，可根据需要添加超时处理
            continue; 
        }

        if (state == 0) {
            if (byte == 0xAA) {
                buffer[0] = byte;
                state = 1;
            }
        } else if (state < 7) {
            buffer[state] = byte;
            state++;
        } else if (state == 7) {
            if (byte == 0x55) {
                buffer[7] = byte;
                return 0; // 帧接收成功
            } else {
                // 帧尾错误，重置状态，重新寻找帧头
                state = 0; 
                return -1;
            }
        }
    }
    return -1;
}

// 解析数据
void parse_data(uint8_t *buffer, SensorData *data) {
    // 航向：Byte1(高) Byte2(低)
    int16_t yaw_raw = (buffer[1] << 8) | buffer[2];
    // 俯仰：Byte3(高) Byte4(低)
    int16_t pitch_raw = (buffer[3] << 8) | buffer[4];
    // 横滚：Byte5(高) Byte6(低)
    int16_t roll_raw = (buffer[5] << 8) | buffer[6];

    data->yaw = yaw_raw / 100.0;
    data->pitch = pitch_raw / 100.0;
    data->roll = roll_raw / 100.0;
}

// 打印数据
void print_data(SensorData *data) {
    printf("\r航向：%6.2f° | 俯仰：%6.2f° | 横滚：%6.2f°", data->yaw, data->pitch, data->roll);
    fflush(stdout);
}

// 显示菜单
void show_menu() {
    printf("\n\n========== 传感器控制菜单 ==========\n");
    printf("1. 查询模式 (发送一次读一次)\n");
    printf("2. 自动模式 (二进制输出)\n");
    printf("3. 自动模式 (ASCII 输出)\n");
    printf("4. 校正俯仰/横滚 0 度 (请保持水平)\n");
    printf("5. 校正航向 0 度\n");
    printf("6. 连续读取数据 (配合自动模式)\n");
    printf("0. 退出\n");
    printf("====================================\n");
    printf("请选择操作：");
}



int imu_init() {
    if (serial_init() < 0) {
        return 1;
    }

    printf("初始化设置为自动模式...\n");
    send_command(0xA5, 0x52);
}


int imu_calib() {
    printf("发送校正俯仰横滚角命令...\n");
    send_command(0xA5, 0x54);
    printf("发送校正航向命令...\n");
    send_command(0xA5, 0x55);

    // 再发送自动模式指令
    send_command(0xA5, 0x52);

    tcflush(fd, TCIFLUSH);
}


int imu_close() {
    close(fd);
    printf("串口已关闭。\n");
}


int imu_read_angle(float *pitch, float *roll, float *yaw) {
    uint8_t rx_buffer[8];
    SensorData sensor;

    // send_command(0xA5, 0x51);
    if (read_frame(rx_buffer) == 0) {
        parse_data(rx_buffer, &sensor);
        *pitch = sensor.pitch;
        *roll = sensor.roll;
        *yaw = sensor.yaw;
        return 0;
    }
    else {
        // printf("IMU读取超时或数据校验错误！\n");
        return -1;
    }
}



int ___main() {
    uint8_t rx_buffer[8];
    SensorData sensor;
    int choice;
    int running = 1;

    // 注册信号处理，以便 Ctrl+C 退出时关闭串口
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    imu_init();

    float pitch = 0.0f;
    float roll = 0.0f;
    float yaw = 0.0f;

    while (running) {
        show_menu();
        if (scanf("%d", &choice) != 1) {
            // 防止输入非数字导致死循环
            while(getchar() != '\n'); 
            continue;
        }

        switch (choice) {
            case 1: // 查询模式
                imu_read_angle(&pitch, &roll, &yaw);
                printf("\r航向：%6.2f° | 俯仰：%6.2f° | 横滚：%6.2f°", yaw, pitch, roll);
                fflush(stdout);
                break;

            case 2: // 自动二进制
                printf("已切换至自动模式 (二进制)。\n");
                send_command(0xA5, 0x52);
                break;

            case 3: // 自动 ASCII
                printf("已切换至自动模式 (ASCII)。\n");
                printf("注意：此模式下本程序解析可能失效，建议直接用串口助手查看。\n");
                send_command(0xA5, 0x53);
                break;

            case 4: // 校正俯仰横滚
            case 5: // 校正航向
                imu_calib();
                break;

            case 6: // 连续读取 (需先切到自动模式)
                printf("开始连续读取 (按 Ctrl+C 停止)...\n");
                while (1) {
                    imu_read_angle(&pitch, &roll, &yaw);
                    printf("\r航向：%6.2f° | 俯仰：%6.2f° | 横滚：%6.2f°", yaw, pitch, roll);
                    fflush(stdout);
                    usleep(50000); // 限制刷新率，避免刷屏太快
                }
                break;

            case 0:
                running = 0;
                break;

            default:
                printf("无效输入。\n");
                break;
        }
    }

    imu_close();

    return 0;
}
