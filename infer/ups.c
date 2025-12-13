#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <time.h>
#include <string.h>

#include "ups.h"

static int ups_fd;

int ups_init() {
    ups_fd = open(I2C_DEVFILE, O_RDWR);
    if(ups_fd < 0) {
        return -1;
    }
    if(ioctl(ups_fd, I2C_SLAVE, UPS_I2C_ADDR) < 0) {
        return -1;
    }
    return 0;
}

// 读取电压寄存器(mV)
int32_t read_ups_voltage() {
    uint16_t data;
    char buf[2];
    
    // 设置寄存器地址为2 (电压寄存器)
    buf[0] = 2;
    if (write(ups_fd, buf, 1) != 1) {
        perror("Failed to set voltage register");
        return -1;
    }
    
    // 读取数据
    if (read(ups_fd, buf, 2) != 2) {
        perror("Failed to read voltage data");
        return -1;
    }
    
    // 字节顺序转换
    data = (buf[0] << 8) | buf[1];
    
    // 计算电压值
    return (int32_t)(data * 1.25 / 1000.0 / 16.0 * 1000.0);
}

// 读取电池容量寄存器
int32_t read_ups_soc() {
    uint16_t data;
    char buf[2];
    
    // 设置寄存器地址为4 (容量寄存器)
    buf[0] = 4;
    if (write(ups_fd, buf, 1) != 1) {
        perror("Failed to set capacity register");
        return -1;
    }
    
    // 读取数据
    if (read(ups_fd, buf, 2) != 2) {
        perror("Failed to read capacity data");
        return -1;
    }
    
    // 字节顺序转换
    data = (buf[0] << 8) | buf[1];
    
    // 计算容量百分比
    return data / 256;
}
