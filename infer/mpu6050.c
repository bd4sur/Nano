/*
 * MPU6050 FIFO Demo - Improved Version
 * 修复: 补码转换、I2C操作规范、FIFO溢出处理、代码结构
 * 平台: Raspberry Pi / Linux with i2c-dev
 * 
 * Compile: gcc -o mpu6050_fifo mpu6050_fifo.c -lm
 * Run: sudo ./mpu6050_fifo
 */

#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <time.h>

/* ============ 配置参数 ============ */
#define MPU6050_I2C_ADDR    0x68
#define I2C_ADAPTER_NR      7           // Raspberry Pi: /dev/i2c-1
#define FIFO_FRAME_SIZE     14          // 完整帧: 6轴+温度 = 14字节
#define FIFO_BUFFER_SIZE    1024        // MPU6050 FIFO大小
#define SAMPLE_RATE_HZ      100         // 目标采样率
#define READ_TIMEOUT_MS     50          // 读取超时(防死锁)

/* ============ 寄存器定义 ============ */
#define REG_ACCEL_XOUT_H    0x3B
#define REG_TEMP_OUT_H      0x41
#define REG_GYRO_XOUT_H     0x43
#define REG_PWR_MGMT_1      0x6B
#define REG_ACCEL_CONFIG    0x1C
#define REG_GYRO_CONFIG     0x1B
#define REG_SMPRT_DIV       0x19
#define REG_CONFIG          0x1A
#define REG_FIFO_EN         0x23
#define REG_USER_CTRL       0x6A
#define REG_FIFO_COUNT_H    0x72
#define REG_FIFO_COUNT_L    0x73
#define REG_FIFO_R_W        0x74
#define REG_WHO_AM_I        0x75

/* ============ 全局变量 ============ */
static int i2c_fd = -1;
static volatile int running = 1;

/* ============ 信号处理(优雅退出) ============ */
#include <signal.h>
void signal_handler(int sig) {
    (void)sig;
    running = 0;
    printf("\n[INFO] Received signal, exiting...\n");
}

/* ============ 规范I2C操作 ============ */

/**
 * @brief 使用i2c_msg实现原子读写操作（Linux推荐方式）
 * @param addr I2C设备地址
 * @param reg 寄存器地址
 * @param data 数据缓冲区
 * @param len 读取字节数
 * @return 成功返回0，失败返回-1
 */
static int i2c_read_bytes(uint8_t addr, uint8_t reg, uint8_t *data, uint8_t len) {
    if (i2c_fd < 0 || !data) return -1;
    
    struct i2c_rdwr_ioctl_data packets;
    struct i2c_msg messages[2];
    
    // 消息1: 写入寄存器地址
    messages[0].addr = addr;
    messages[0].flags = 0;  // 写操作
    messages[0].len = 1;
    messages[0].buf = &reg;
    
    // 消息2: 读取数据（地址自动递增）
    messages[1].addr = addr;
    messages[1].flags = I2C_M_RD;  // 读操作
    messages[1].len = len;
    messages[1].buf = data;
    
    packets.msgs = messages;
    packets.nmsgs = 2;
    
    if (ioctl(i2c_fd, I2C_RDWR, &packets) < 0) {
        fprintf(stderr, "[ERROR] I2C read failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

/**
 * @brief 单字节写入（配置寄存器用）
 */
static int i2c_write_byte(uint8_t addr, uint8_t reg, uint8_t value) {
    if (i2c_fd < 0) return -1;
    
    uint8_t buf[2] = {reg, value};
    
    struct i2c_msg msg = {
        .addr = addr,
        .flags = 0,
        .len = 2,
        .buf = buf
    };
    
    struct i2c_rdwr_ioctl_data packets = {
        .msgs = &msg,
        .nmsgs = 1
    };
    
    if (ioctl(i2c_fd, I2C_RDWR, &packets) < 0) {
        fprintf(stderr, "[ERROR] I2C write failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

/* ============ 数据转换工具 ============ */

/**
 * @brief 合并高低字节为16位有符号整数（✅ 修复补码转换）
 * @param msb 高字节
 * @param lsb 低字节
 * @return int16_t 有符号结果
 * 
 * 💡 原理: C语言中int16_t自动处理二进制补码，
 *    直接强制转换即可，无需手动判断符号位
 */
static inline int16_t merge_int16(uint8_t msb, uint8_t lsb) {
    return (int16_t)((msb << 8) | lsb);
}

/**
 * @brief 加速度计原始值转g单位
 * @param raw 原始ADC值
 * @param accel_scale 量程配置(0:±2g, 1:±4g, 2:±8g, 3:±16g)
 */
static float accel_raw_to_g(int16_t raw, uint8_t accel_scale) {
    static const float sensitivity[] = {16384.0f, 8192.0f, 4096.0f, 2048.0f};
    return (float)raw / sensitivity[accel_scale & 0x03];
}

/**
 * @brief 陀螺仪原始值转°/s
 * @param raw 原始ADC值
 * @param gyro_scale 量程配置(0:±250, 1:±500, 2:±1000, 3:±2000 °/s)
 */
static float gyro_raw_to_dps(int16_t raw, uint8_t gyro_scale) {
    static const float sensitivity[] = {131.0f, 65.5f, 32.8f, 16.4f};
    return (float)raw / sensitivity[gyro_scale & 0x03];
}

/**
 * @brief 温度原始值转摄氏度
 * 公式: Temp(°C) = (TEMP_out / 340) + 36.53
 */
static float temp_raw_to_celsius(int16_t raw) {
    return ((float)raw / 340.0f) + 36.53f;
}

/**
 * @brief 获取单调递增时间戳（毫秒）
 * ✅ 替代 clock()，精度达纳秒级，不受系统时间调整影响
 */
static uint32_t get_timestamp_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) < 0) {
        return 0;  // 错误时返回0
    }
    return (uint32_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

/* ============ MPU6050初始化 ============ */

static int _mpu6050_init(void) {
    uint8_t who_am_i;
    
    // 1. 读取WHO_AM_I验证连接
    if (i2c_read_bytes(MPU6050_I2C_ADDR, REG_WHO_AM_I, &who_am_i, 1) < 0) {
        return -1;
    }
    if (who_am_i != 0x68 && who_am_i != 0x69) {
        fprintf(stderr, "[ERROR] Invalid WHO_AM_I: 0x%02X\n", who_am_i);
        return -1;
    }
    printf("[INFO] MPU6050 detected (WHO_AM_I=0x%02X)\n", who_am_i);
    
    // 2. 唤醒设备
    if (i2c_write_byte(MPU6050_I2C_ADDR, REG_PWR_MGMT_1, 0x01) < 0) return -1;
    usleep(10000);  // 等待PLL稳定
    
    // 3. 配置加速度计: ±2g, 带宽44.8Hz
    if (i2c_write_byte(MPU6050_I2C_ADDR, REG_ACCEL_CONFIG, 0x00) < 0) return -1;
    
    // 4. 配置陀螺仪: ±250°/s, 带宽42Hz
    if (i2c_write_byte(MPU6050_I2C_ADDR, REG_GYRO_CONFIG, 0x00) < 0) return -1;
    
    // 5. 配置采样率: 1kHz / (7+1) = 125Hz
    //    SMPRT_DIV = (陀螺输出率/目标采样率) - 1
    if (i2c_write_byte(MPU6050_I2C_ADDR, REG_SMPRT_DIV, 0x07) < 0) return -1;
    
    // 6. 配置DLPF: 带宽42Hz (降低噪声)
    if (i2c_write_byte(MPU6050_I2C_ADDR, REG_CONFIG, 0x03) < 0) return -1;
    
    // 7. 启用FIFO: 温度+加速度+陀螺仪
    if (i2c_write_byte(MPU6050_I2C_ADDR, REG_FIFO_EN, 0xF8) < 0) return -1;
    // 0xF8 = 0b11111000: TEMP|XG|YG|ZG|ACCEL FIFO enabled
    
    // 8. 启用FIFO并重置指针
    if (i2c_write_byte(MPU6050_I2C_ADDR, REG_USER_CTRL, 0x44) < 0) return -1;
    // 0x44: FIFO_EN=1, FIFO_RESET=1
    
    usleep(50000);  // 等待配置生效
    return 0;
}

/* ============ FIFO数据读取与解析 ============ */

typedef struct {
    int16_t ax, ay, az;      // 加速度计 (raw)
    int16_t gx, gy, gz;      // 陀螺仪 (raw)
    int16_t temp;             // 温度 (raw)
    float ax_g, ay_g, az_g;  // 加速度计 (g)
    float gx_dps, gy_dps, gz_dps;  // 陀螺仪 (°/s)
    float temp_c;             // 温度 (°C)
    uint32_t timestamp_ms;    // 时间戳
} mpu6050_frame_t;

/**
 * @brief 从FIFO读取并解析一帧数据
 * @param frame 输出结构体指针
 * @return 成功返回0，FIFO空返回-1，错误返回-2
 */
static int fifo_read_frame(mpu6050_frame_t *frame, uint8_t accel_scale, uint8_t gyro_scale) {
    uint8_t buffer[FIFO_FRAME_SIZE];
    uint16_t fifo_count;
    
    // 1. 读取FIFO字节计数
    uint8_t count_buf[2];
    if (i2c_read_bytes(MPU6050_I2C_ADDR, REG_FIFO_COUNT_H, count_buf, 2) < 0) {
        return -2;
    }
    fifo_count = ((uint16_t)count_buf[0] << 8) | count_buf[1];
    
    // 2. 检查数据是否足够
    if (fifo_count < FIFO_FRAME_SIZE) {
        return -1;  // FIFO空，调用方延时后重试
    }
    
    // 3. 检查FIFO溢出
    if (fifo_count >= FIFO_BUFFER_SIZE - FIFO_FRAME_SIZE) {
        fprintf(stderr, "[WARN] FIFO nearly full (%d bytes), resetting...\n", fifo_count);
        // 重置FIFO指针
        i2c_write_byte(MPU6050_I2C_ADDR, REG_USER_CTRL, 0x44);
        return -2;
    }
    
    // 4. 批量读取一帧数据 (FIFO地址自动递增)
    if (i2c_read_bytes(MPU6050_I2C_ADDR, REG_FIFO_R_W, buffer, FIFO_FRAME_SIZE) < 0) {
        return -2;
    }
    
    // 5. 解析数据 (MPU6050 FIFO顺序: AX,AY,AZ,TEMP,GX,GY,GZ)
    frame->ax = merge_int16(buffer[0], buffer[1]);
    frame->ay = merge_int16(buffer[2], buffer[3]);
    frame->az = merge_int16(buffer[4], buffer[5]);
    frame->temp = merge_int16(buffer[6], buffer[7]);
    frame->gx = merge_int16(buffer[8], buffer[9]);
    frame->gy = merge_int16(buffer[10], buffer[11]);
    frame->gz = merge_int16(buffer[12], buffer[13]);
    
    // 6. 单位转换
    frame->ax_g = accel_raw_to_g(frame->ax, accel_scale);
    frame->ay_g = accel_raw_to_g(frame->ay, accel_scale);
    frame->az_g = accel_raw_to_g(frame->az, accel_scale);
    
    frame->gx_dps = gyro_raw_to_dps(frame->gx, gyro_scale);
    frame->gy_dps = gyro_raw_to_dps(frame->gy, gyro_scale);
    frame->gz_dps = gyro_raw_to_dps(frame->gz, gyro_scale);
    
    frame->temp_c = temp_raw_to_celsius(frame->temp);
    
    // 7. 时间戳
    static uint32_t start_ms = 0;
    if (start_ms == 0) {
        start_ms = get_timestamp_ms();
    }
    frame->timestamp_ms = get_timestamp_ms() - start_ms;
    
    return 0;
}

/* ============ 姿态角计算（加速度计静态解算） ============ */

/**
 * @brief 使用加速度计计算俯仰角和滚转角（静态场景）
 * @param ax,ay,az 加速度计数据(g)
 * @param pitch,roll 输出角度(度)
 */
static void calc_pitch_roll(float ax, float ay, float az, float *pitch, float *roll) {
    // 防止除零
    float horiz = sqrtf(ay*ay + az*az);
    if (horiz < 0.001f) horiz = 0.001f;
    
    *pitch = atan2f(-ax, horiz) * 180.0f / M_PI;
    *roll  = atan2f(ay, az) * 180.0f / M_PI;
}

/* ============ 主函数 ============ */


int mpu6050_init() {
    char bus_path[64];
    // 1. 打开I2C总线
    snprintf(bus_path, sizeof(bus_path), "/dev/i2c-%d", I2C_ADAPTER_NR);
    i2c_fd = open(bus_path, O_RDWR);
    if (i2c_fd < 0) {
        fprintf(stderr, "[ERROR] Cannot open %s: %s\n", bus_path, strerror(errno));
        fprintf(stderr, "[HINT] Try: sudo modprobe i2c-dev && sudo ./mpu6050_fifo\n");
        return EXIT_FAILURE;
    }
    printf("[INFO] Opened %s\n", bus_path);
    
    // 2. 初始化MPU6050
    if (_mpu6050_init() < 0) {
        fprintf(stderr, "[ERROR] MPU6050 initialization failed\n");
        close(i2c_fd);
        return EXIT_FAILURE;
    }
}



int mpu6050_read_angle(float *pitch, float *roll, float *yaw, float *temp) {
    mpu6050_frame_t frame;
    uint8_t accel_scale = 0;  // ±2g
    uint8_t gyro_scale = 0;   // ±250°/s
    *pitch = 0.0f;
    *roll = 0.0f;
    *yaw = 0.0f;
    *temp = 0.0f;

    int ret = fifo_read_frame(&frame, accel_scale, gyro_scale);

    if (ret == 0) {
        calc_pitch_roll(frame.ax_g, frame.ay_g, frame.az_g, pitch, roll);
        *temp = frame.temp_c;
    }

    return ret;
}


int mpu6050_close() {
    // 4. 清理资源
    printf("\n[INFO] Cleaning up...\n");
    
    // 禁用FIFO
    i2c_write_byte(MPU6050_I2C_ADDR, REG_USER_CTRL, 0x00);
    i2c_write_byte(MPU6050_I2C_ADDR, REG_FIFO_EN, 0x00);
    
    // 关闭I2C
    if (i2c_fd >= 0) {
        close(i2c_fd);
    }
    
    printf("[INFO] Exit successfully.\n");
}



int __main(int argc, char *argv[]) {

    float pitch = 0.0f;
    float roll = 0.0f;
    float yaw = 0.0f;
    float temp = 0.0f;
    uint32_t last_print_ms = 0;
    
    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    mpu6050_init();
    
    printf("[INFO] Starting FIFO read loop (Ctrl+C to exit)...\n");
    printf("%-10s %-10s %-10s %-8s\n", 
           "Pitch(°)", "Roll(°)", "Yaw(°)", "Temp(°C)");
    printf("--------------------------------------------------------------------------------\n");
    
    // 3. 主循环
    while (running) {

        int ret = mpu6050_read_angle(&pitch, &roll, &yaw, &temp);

        if (ret == 0) {
            printf("%-10.2f %-10.2f %-10.2f %-10.2f\n", pitch, roll, yaw, temp);
        }

    
    }
    
    mpu6050_close();

    return EXIT_SUCCESS;
}
