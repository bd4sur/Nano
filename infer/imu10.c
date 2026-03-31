#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <sys/ioctl.h>

#include "imu.h"

/* 配置项 / Config */
#ifndef IMU_UART_RX_BUF_SIZE
#define IMU_UART_RX_BUF_SIZE 512 /* 环形接收缓冲区大小 / RX ring buffer size */
#endif

#define FRAME_HEAD1 0x7E
#define FRAME_HEAD2 0x23

/* 功能码 / Function codes */
#define IMU_FUNC_VERSION 0x01
#define IMU_FUNC_RAW_ACCEL 0x04
#define IMU_FUNC_RAW_GYRO 0x0A
#define IMU_FUNC_RAW_MAG 0x10
#define IMU_FUNC_QUAT 0x16
#define IMU_FUNC_EULER 0x26
#define IMU_FUNC_BARO 0x32
#define IMU_FUNC_SET_FREQ 0x60
#define IMU_FUNC_CALIB_IMU 0x70
#define IMU_FUNC_CALIB_MAG 0x71
#define IMU_FUNC_CALIB_BARO 0x72
#define IMU_FUNC_CALIB_TEMP 0x73
#define IMU_FUNC_REQUEST_DATA 0x80
#define IMU_FUNC_RETURN_STATE 0x81
#define IMU_FUNC_RESET_FLASH 0xA0

/* 结构体：一次性获取所有传感器数据 / Struct: get all sensor data at once */
typedef struct
{
    float accel[3];
    float gyro[3];
    float mag[3];
    float quat[4];
    float euler[3];
    float baro[4];
    char version[8];
} imu_measurement_t;

#define USART_BAUDRATE B115200
#define USART_BAUDRATE_VAL 115200

/* Linux文件描述符替代Arduino的Serial对象 */
int mySerialFd;

/* 配置串口参数（波特率、8N1、原始模式、非阻塞） */
static int _serial_config(int fd, speed_t baud)
{
    struct termios options;

    if (fd < 0)
        return -1;

    /* 获取当前串口配置 */
    if (tcgetattr(fd, &options) != 0)
    {
        perror("tcgetattr failed");
        return -1;
    }

    /* 设置波特率 */
    cfsetispeed(&options, baud);
    cfsetospeed(&options, baud);

    /* 设置数据格式：8数据位、无校验、1停止位 */
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag |= CLOCAL | CREAD;

    /* 禁用软件流控 */
    options.c_iflag &= ~(IXON | IXOFF | IXANY);

    /* 禁用硬件流控 */
    options.c_cflag &= ~CRTSCTS;

    /* 设置原始输入模式（禁用特殊字符处理） */
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

    /* 设置原始输出模式 */
    options.c_oflag &= ~OPOST;

    /* 设置超时：非阻塞读取 */
    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 0;

    /* 应用配置 */
    if (tcsetattr(fd, TCSANOW, &options) != 0)
    {
        perror("tcsetattr failed");
        return -1;
    }

    /* 清空缓冲区 */
    tcflush(fd, TCIFLUSH);

    return 0;
}

/* 模拟Arduino的available()：返回可读取字节数 */
static int _serial_available(int fd)
{
    int count = 0;
    if (fd >= 0 && ioctl(fd, FIONREAD, &count) == 0)
    {
        return count;
    }
    return 0;
}

/* 批量读取串口数据到缓冲区，返回实际读取的字节数 */
static ssize_t _serial_read_batch(int fd, uint8_t *buffer, size_t max_len)
{
    if (fd < 0 || !buffer || max_len == 0)
        return -1;
    
    int available = _serial_available(fd);
    if (available <= 0)
        return 0;
    
    /* 限制读取数量不超过缓冲区大小 */
    size_t to_read = (available > (int)max_len) ? max_len : (size_t)available;
    ssize_t n = read(fd, buffer, to_read);
    return n;
}

/* ================= 公开API实现 ================= */

/* 串口初始化 */
static inline void uart_init(void)
{
    mySerialFd = open(IMU_DEVFILE, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (mySerialFd < 0)
    {
        fprintf(stderr, "Error: Cannot open serial port %s: %s\n", IMU_DEVFILE, strerror(errno));
        exit(EXIT_FAILURE);
    }
    if (_serial_config(mySerialFd, USART_BAUDRATE) < 0)
    {
        fprintf(stderr, "Error: Failed to configure serial port\n");
        exit(EXIT_FAILURE);
    }
}

/* 发送单个字节 */
inline void uart_send_uint8(uint8_t Data)
{
    if (mySerialFd >= 0)
    {
        write(mySerialFd, &Data, 1);
    }
}

/* 发送字节数组 */
inline void uart_send_uint8_array(uint8_t *pData, uint16_t Length)
{
    if (mySerialFd >= 0 && pData != NULL && Length > 0)
    {
        write(mySerialFd, pData, Length);
    }
}

/* 接收处理：读取可用数据并传递给IMU驱动 */
int32_t uart_receive(uint8_t *rxbyte)
{
    if (mySerialFd >= 0 && _serial_available(mySerialFd) > 0)
    {
        ssize_t n = read(mySerialFd, rxbyte, 1);
        if (n == 1)
        {
            return 0;
        }
    }
    return -1;
}

/* ---------- 环形缓冲 / RX ring buffer ---------- */
static volatile uint8_t s_rx_buffer[IMU_UART_RX_BUF_SIZE];
static volatile uint16_t s_rx_write_index = 0; /* 写入位置 / write index */
static volatile uint16_t s_rx_read_index = 0;  /* 读取位置 / read index */

/* ---------- 缓冲区管理 / Buffer management ---------- */

/**
 * @brief 清空环形缓冲区
 *        Clear/reset the RX ring buffer
 */
static inline void _rxbuf_clear(void)
{
    s_rx_write_index = 0;
    s_rx_read_index = 0;
}

/**
 * @brief 清空串口硬件缓冲区
 *        Clear the serial port hardware buffer
 */
static inline void _serial_flush(int fd)
{
    if (fd >= 0)
    {
        tcflush(fd, TCIFLUSH);
    }
}

/**
 * @brief 清空所有缓冲区（串口硬件缓冲区 + 环形缓冲区）
 *        Clear all buffers to discard stale data
 */
static void _discard_stale_data(void)
{
    _serial_flush(mySerialFd);
    _rxbuf_clear();
}

static inline uint16_t _rxbuf_next(uint16_t index)
{
    return (uint16_t)((index + 1u) % IMU_UART_RX_BUF_SIZE);
}

static inline int _rxbuf_is_empty(void)
{
    return s_rx_write_index == s_rx_read_index;
}

static inline void _rxbuf_push(uint8_t byte_value)
{
    uint16_t next_index = _rxbuf_next(s_rx_write_index);
    if (next_index == s_rx_read_index)
    {
        /* 缓冲区满时丢弃最旧数据 / drop oldest byte when buffer is full */
        s_rx_read_index = _rxbuf_next(s_rx_read_index);
    }
    s_rx_buffer[s_rx_write_index] = byte_value;
    s_rx_write_index = next_index;
}

static inline int _rxbuf_pop(uint8_t *out_byte)
{
    if (_rxbuf_is_empty())
    {
        return -1;
    }
    *out_byte = s_rx_buffer[s_rx_read_index];
    s_rx_read_index = _rxbuf_next(s_rx_read_index);
    return 0;
}

/** 将两个字节转换为 int16 / Convert two bytes to int16 */
static int16_t to_int16(const uint8_t *bytes)
{
    return (int16_t)((bytes[1] << 8) + bytes[0]);
}

/** 将四个字节转换为 float / Convert four bytes to float */
static float to_float(const uint8_t *bytes)
{
    float v;
    memcpy(&v, bytes, sizeof(float));
    return v;
}

/* ---------- 内部状态/ Internal cached state ---------- */
static volatile float s_ax = 0.0f, s_ay = 0.0f, s_az = 0.0f;
static volatile float s_gx = 0.0f, s_gy = 0.0f, s_gz = 0.0f;
static volatile float s_mx = 0.0f, s_my = 0.0f, s_mz = 0.0f;
static volatile float s_roll = 0.0f, s_pitch = 0.0f, s_yaw = 0.0f;
static volatile float s_q0 = 0.0f, s_q1 = 0.0f, s_q2 = 0.0f, s_q3 = 0.0f;
static volatile float s_height = 0.0f, s_temperature = 0.0f, s_pressure = 0.0f, s_pressure_contrast = 0.0f;
static volatile int s_version_high = -1, s_version_mid = 0, s_version_low = 0;
static volatile uint8_t s_last_rx_function = 0;
static volatile int16_t s_last_rx_state = 0;

/* ---------- 解析数据帧 / Parse one complete frame ---------- */
static void _parse_frame_data(uint8_t frame_function, const uint8_t *frame_data)
{
    if (frame_function == IMU_FUNC_RAW_ACCEL)
    {
        float accel_ratio = 16.0f / 32767.0f;
        s_ax = to_int16(&frame_data[0]) * accel_ratio;
        s_ay = to_int16(&frame_data[2]) * accel_ratio;
        s_az = to_int16(&frame_data[4]) * accel_ratio;

        float deg_to_rad = 3.14159265358979323846f / 180.0f;
        float gyro_ratio = (2000.0f / 32767.0f) * deg_to_rad;
        s_gx = to_int16(&frame_data[6]) * gyro_ratio;
        s_gy = to_int16(&frame_data[8]) * gyro_ratio;
        s_gz = to_int16(&frame_data[10]) * gyro_ratio;

        float mag_ratio = 800.0f / 32767.0f;
        s_mx = to_int16(&frame_data[12]) * mag_ratio;
        s_my = to_int16(&frame_data[14]) * mag_ratio;
        s_mz = to_int16(&frame_data[16]) * mag_ratio;
    }
    else if (frame_function == IMU_FUNC_EULER)
    {
        s_roll = to_float(&frame_data[0]);
        s_pitch = to_float(&frame_data[4]);
        s_yaw = to_float(&frame_data[8]);
    }
    else if (frame_function == IMU_FUNC_QUAT)
    {
        s_q0 = to_float(&frame_data[0]);
        s_q1 = to_float(&frame_data[4]);
        s_q2 = to_float(&frame_data[8]);
        s_q3 = to_float(&frame_data[12]);
    }
    else if (frame_function == IMU_FUNC_BARO)
    {
        s_height = to_float(&frame_data[0]);
        s_temperature = to_float(&frame_data[4]);
        s_pressure = to_float(&frame_data[8]);
        s_pressure_contrast = to_float(&frame_data[12]);
    }
    else if (frame_function == IMU_FUNC_VERSION)
    {
        s_version_high = frame_data[0];
        s_version_mid = frame_data[1];
        s_version_low = frame_data[2];
    }
    else if (frame_function == IMU_FUNC_RETURN_STATE)
    {
        s_last_rx_function = frame_data[0];
        s_last_rx_state = (int16_t)frame_data[1];
    }
}

/* ---------- 帧发送接口 / Command sender ---------- */

void Send_IMU_Array(uint8_t *pData, uint8_t Length)
{
    uart_send_uint8_array(pData, Length);
}

int IMU_UART_SendCommand(uint8_t function, const uint8_t *params, uint8_t param_len)
{
    if (param_len > 3 || (param_len > 0 && params == NULL))
    {
        return -1;
    }

    uint8_t frame[8] = {FRAME_HEAD1, FRAME_HEAD2, 0, function, 0, 0, 0, 0};

    for (uint8_t i = 0; i < param_len; ++i)
    {
        frame[4 + i] = params[i];
    }

    uint8_t frame_len = (uint8_t)(4 + param_len + 1);
    frame[2] = frame_len;

    uint8_t checksum = 0;
    for (uint8_t i = 0; i < frame_len - 1; ++i)
    {
        checksum = (uint8_t)(checksum + frame[i]);
    }
    frame[frame_len - 1] = checksum;

    Send_IMU_Array(frame, frame_len);
    return 0;
}

/** 初始化接口，当前为占位符 / Init hook (placeholder). */
void IMU_UART_Init(void)
{
}

/**
 * @brief 中断接收入口，将新数据写入环形缓冲
 *        ISR entry to push received bytes into ring buffer
 */
void IMU_UART_RxBytes(const uint8_t *data, uint16_t len)
{
    if (!data || len == 0)
        return;
    for (uint16_t i = 0; i < len; ++i)
    {
        _rxbuf_push(data[i]);
    }
}

/**
 * @brief 批量从串口读取数据并推入环形缓冲区
 *        Read data from serial port in batch and push to ring buffer
 * @return 实际读取并推入环形缓冲区的字节数
 */
static uint16_t _uart_receive_batch(void)
{
    uint8_t temp_buffer[256];
    ssize_t n = _serial_read_batch(mySerialFd, temp_buffer, sizeof(temp_buffer));
    if (n > 0)
    {
        IMU_UART_RxBytes(temp_buffer, (uint16_t)n);
        return (uint16_t)n;
    }
    return 0;
}

/**
 * @brief 解析环形缓冲中的数据，提取完整帧并更新缓存
 *        Process RX ring buffer, parse frames and update internal cache
 */
void IMU_UART_Process(void)
{
    enum
    {
        RX_STATE_EXPECT_HEAD1 = 0,
        RX_STATE_EXPECT_HEAD2,
        RX_STATE_EXPECT_LENGTH,
        RX_STATE_EXPECT_FUNCTION,
        RX_STATE_COLLECT_DATA
    };

    static uint8_t rx_state = RX_STATE_EXPECT_HEAD1;
    static uint8_t frame_length = 0;
    static uint8_t frame_function = 0;
    static uint8_t frame_buffer[64]; /* 数据区 + 校验 / data section + checksum */
    static uint16_t frame_index = 0;

    uint8_t current_byte = 0;

    while (_rxbuf_pop(&current_byte) == 0)
    {
        switch (rx_state)
        {
        case RX_STATE_EXPECT_HEAD1:
            rx_state = (current_byte == FRAME_HEAD1) ? RX_STATE_EXPECT_HEAD2 : RX_STATE_EXPECT_HEAD1;
            break;

        case RX_STATE_EXPECT_HEAD2:
            rx_state = (current_byte == FRAME_HEAD2) ? RX_STATE_EXPECT_LENGTH : RX_STATE_EXPECT_HEAD1;
            break;

        case RX_STATE_EXPECT_LENGTH:
            frame_length = current_byte;
            rx_state = RX_STATE_EXPECT_FUNCTION;
            break;

        case RX_STATE_EXPECT_FUNCTION:
            frame_function = current_byte;
            frame_index = 0;
            rx_state = RX_STATE_COLLECT_DATA;
            break;

        case RX_STATE_COLLECT_DATA:
        {
            uint16_t data_length = (frame_length >= 4) ? (uint16_t)(frame_length - 4) : 0;
            if (data_length == 0 || data_length > sizeof(frame_buffer))
            {
                rx_state = RX_STATE_EXPECT_HEAD1;
                break;
            }

            frame_buffer[frame_index++] = current_byte;
            if (frame_index >= data_length)
            {
                uint8_t calculated_checksum = (uint8_t)(FRAME_HEAD1 + FRAME_HEAD2 + frame_length + frame_function);
                for (uint16_t i = 0; i < data_length - 1; ++i)
                {
                    calculated_checksum = (uint8_t)(calculated_checksum + frame_buffer[i]);
                }

                uint8_t received_checksum = frame_buffer[data_length - 1];
                if (calculated_checksum == received_checksum)
                {
                    _parse_frame_data(frame_function, frame_buffer);
                }
                rx_state = RX_STATE_EXPECT_HEAD1;
            }
        }
        break;

        default:
            rx_state = RX_STATE_EXPECT_HEAD1;
            break;
        }
    }
}

/* ---------------- 读取数据 / Read Data ---------------- */
int IMU_UART_GetAccelerometer(float out[3])
{
    if (!out)
        return -1;
    out[0] = s_ax;
    out[1] = s_ay;
    out[2] = s_az;
    return 0;
}

/**
 * @brief 读取角速度数据（rad/s）
 *        Read angular velocity in rad/s.
 */
int IMU_UART_GetGyroscope(float out[3])
{
    if (!out)
        return -1;
    out[0] = s_gx;
    out[1] = s_gy;
    out[2] = s_gz;
    return 0;
}

/**
 * @brief 读取磁场数据（uT）
 *        Read magnetic field in micro tesla.
 */
int IMU_UART_GetMagnetometer(float out[3])
{
    if (!out)
        return -1;
    out[0] = s_mx;
    out[1] = s_my;
    out[2] = s_mz;
    return 0;
}

/**
 * @brief 读取四元数
 *        Read quaternion (w, x, y, z).
 */
int IMU_UART_GetQuaternion(float out[4])
{
    if (!out)
        return -1;
    out[0] = s_q0;
    out[1] = s_q1;
    out[2] = s_q2;
    out[3] = s_q3;
    return 0;
}

/**
 * @brief 读取欧拉角（角度）
 *        Read Euler angles in radians.
 */
int IMU_UART_GetEuler(float out[3])
{
    if (!out)
        return -1;
    const float RAD2DEG = 57.2957795f;
    out[0] = s_roll * RAD2DEG;
    out[1] = s_pitch * RAD2DEG;
    out[2] = s_yaw * RAD2DEG;
    return 0;
}

/**
 * @brief 读取气压相关数据：高度、温度、气压、气压差
 *        Read barometric data: height, temperature, pressure, delta.
 */
int IMU_UART_GetBarometer(float out[4])
{
    if (!out)
        return -1;
    out[0] = s_height;
    out[1] = s_temperature;
    out[2] = s_pressure;
    out[3] = s_pressure_contrast;
    return 0;
}

/**
 * @brief 读取固件版本字符串
 *        Read firmware version string.
 */
void IMU_UART_GetVersion(void)
{
    if (s_version_high < 0)
    {
        uint8_t payload[2] = {IMU_FUNC_VERSION, 0x00};
        IMU_UART_SendCommand(IMU_FUNC_REQUEST_DATA, payload, (uint8_t)sizeof(payload));

        for (int i = 0; i < 20; ++i)
        {
            /* 批量读取串口数据到环形缓冲区 */
            _uart_receive_batch();
            IMU_UART_Process();
            if (s_version_high >= 0)
            {
                char buffer[50];
                sprintf(buffer, "Version:%d.%d.%d\r\n", s_version_high, s_version_mid, s_version_low);
                printf("%s\n", buffer);
                return;
            }
            usleep(500);
        }
        printf("Version:-1\n");
        return;
    }
}

/**
 * @brief 一次性读取全部常用数据
 *        Read all common sensor values at once.
 */
int IMU_UART_GetAll(imu_measurement_t *out)
{
    if (!out)
        return -1;
    IMU_UART_GetAccelerometer(out->accel);
    IMU_UART_GetGyroscope(out->gyro);
    IMU_UART_GetMagnetometer(out->mag);
    IMU_UART_GetQuaternion(out->quat);
    IMU_UART_GetEuler(out->euler);
    IMU_UART_GetBarometer(out->baro);
    return 0;
}

/* ---------- 清理缓存 / Clear cached auto-reported data ---------- */
void IMU_UART_ClearAutoReportData(void)
{
    s_ax = s_ay = s_az = 0.0f;
    s_gx = s_gy = s_gz = 0.0f;
    s_mx = s_my = s_mz = 0.0f;
    s_roll = s_pitch = s_yaw = 0.0f;
    s_q0 = s_q1 = s_q2 = s_q3 = 0.0f;
    s_height = s_temperature = s_pressure = s_pressure_contrast = 0.0f;
}


int IMU_UART_WaitCalibration(uint8_t function, uint32_t timeout_ms)
{
    uint32_t elapsed_ms = 0;
    while (1)
    {
        /* 批量读取串口数据到环形缓冲区 */
        _uart_receive_batch();
        IMU_UART_Process();

        if (s_last_rx_function == function)
        {
            return s_last_rx_state;
        }

        if (timeout_ms != 0 && elapsed_ms >= timeout_ms)
        {
            return -1;
        }

        usleep(100);
        if (timeout_ms != 0)
        {
            ++elapsed_ms;
        }
    }
}

static int _calibration_with_wait(uint8_t function, const uint8_t *payload, uint8_t payload_len,
                                  const char *label, uint32_t timeout_ms)
{
    s_last_rx_function = 0;
    s_last_rx_state = -1;

    int rc = IMU_UART_SendCommand(function, payload, payload_len);
    if (rc != 0)
    {
        return rc;
    }

    int result = IMU_UART_WaitCalibration(function, timeout_ms);
    if (!label)
    {
        label = "unknown";
    }

    char buffer[100];

    if (result == -1)
    {
        sprintf(buffer, "[IMU] Calibration %s timeout\r\n", label);
        printf("%s\n", buffer);
    }
    else if (result == 1)
    {
        sprintf(buffer, "[IMU] Calibration %s success\r\n", label);
        printf("%s\n", buffer);
    }
    else
    {
        sprintf(buffer, "[IMU] Calibration %s failed (code=%d)\r\n", label, result);
        printf("%s\n", buffer);
    }

    return result;
}

/*校准API*/
int IMU_UART_CalibrationImu(void)
{
    uint8_t payload[2] = {0x01, 0x5F};
    return _calibration_with_wait(IMU_FUNC_CALIB_IMU, payload, (uint8_t)sizeof(payload), "imu", 7000);
}

int IMU_UART_CalibrationMag(void)
{
    uint8_t payload[2] = {0x01, 0x5F};
    return _calibration_with_wait(IMU_FUNC_CALIB_MAG, payload, (uint8_t)sizeof(payload), "mag", 0);
}

int IMU_UART_CalibrationTemp(float now_temperature)
{
    if (now_temperature > 50.0f || now_temperature < -50.0f)
    {
        return -1;
    }
    int16_t temperature_raw = (int16_t)(now_temperature * 100.0f);
    uint8_t param_low = (uint8_t)(temperature_raw & 0xFF);
    uint8_t param_high = (uint8_t)((temperature_raw >> 8) & 0xFF);
    uint8_t payload[3] = {param_low, param_high, 0x5F};
    return _calibration_with_wait(IMU_FUNC_CALIB_TEMP, payload, (uint8_t)sizeof(payload), "temp", 2000);
}

int IMU_UART_ResetUserData(void)
{
    uint8_t payload[2] = {0x01, 0x5F};
    return IMU_UART_SendCommand(IMU_FUNC_RESET_FLASH, payload, (uint8_t)sizeof(payload));
}


int IMU_UART_SetFreq(uint8_t freq)
{
    uint8_t payload[2] = {0x01, 0x5F};
    payload[0] = freq;
    return IMU_UART_SendCommand(IMU_FUNC_SET_FREQ, payload, (uint8_t)sizeof(payload));
}


void Send_IMU_Data(uint8_t Data)
{
    uart_send_uint8(Data);
}






int imu_init() {
    uart_init();
    IMU_UART_GetVersion();
    IMU_UART_SetFreq(20);
    return 0;
}


int imu_reset() {
    return 0;
}


int imu_calib() {
    return 0;
}


int imu_close() {
    close(mySerialFd);
    printf("串口已关闭。\n");
    return 0;
}


int imu_read_angle(float *pitch, float *roll, float *yaw) {
    imu_measurement_t imu_data;

    /* 
     * 策略：丢弃过时数据，获取最新实时数据
     * Strategy: Discard stale data, get the latest real-time data
     * 
     * 步骤1：批量读取当前所有可用数据到环形缓冲区
     * Step 1: Read all currently available data into ring buffer
     */
    _uart_receive_batch();
    
    /* 
     * 步骤2：处理所有数据帧，最后一帧会覆盖之前的，所以静态变量中是最新数据
     * Step 2: Process all frames, the last one will overwrite previous ones
     */
    IMU_UART_Process();
    
    /* 
     * 步骤3：获取当前最新数据
     * Step 3: Get the latest data
     */
    int ret = IMU_UART_GetAll(&imu_data);

    /* 
     * 步骤4：清空所有缓冲区，丢弃未处理的旧数据，确保下次读取的是新数据
     * Step 4: Clear all buffers to discard unconsumed stale data
     */
    _discard_stale_data();

    if (ret == 0) {
        *pitch = imu_data.euler[1];
        *roll = imu_data.euler[0];
        *yaw = imu_data.euler[2];
    }
    else {
        *pitch = 0.0f;
        *roll = 0.0f;
        *yaw = 0.0f;
    }
    return ret;
}

int imu_read_quaternion(float *q0, float *q1, float *q2, float *q3) {
    imu_measurement_t imu_data;

    /* 
     * 策略：丢弃过时数据，获取最新实时数据
     * Strategy: Discard stale data, get the latest real-time data
     * 
     * 步骤1：批量读取当前所有可用数据到环形缓冲区
     * Step 1: Read all currently available data into ring buffer
     */
    _uart_receive_batch();
    
    /* 
     * 步骤2：处理所有数据帧，最后一帧会覆盖之前的，所以静态变量中是最新数据
     * Step 2: Process all frames, the last one will overwrite previous ones
     */
    IMU_UART_Process();
    
    /* 
     * 步骤3：获取当前最新数据
     * Step 3: Get the latest data
     */
    int ret = IMU_UART_GetAll(&imu_data);

    /* 
     * 步骤4：清空所有缓冲区，丢弃未处理的旧数据，确保下次读取的是新数据
     * Step 4: Clear all buffers to discard unconsumed stale data
     */
    _discard_stale_data();

    if (ret == 0) {
        *q0 = imu_data.quat[0];
        *q1 = imu_data.quat[1];
        *q2 = imu_data.quat[2];
        *q3 = imu_data.quat[3];
    }
    else {
        *q0 = 0.0f;
        *q1 = 0.0f;
        *q2 = 0.0f;
        *q3 = 0.0f;
    }
    return ret;
}
