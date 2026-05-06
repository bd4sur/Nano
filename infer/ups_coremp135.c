#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <math.h>

#include "ups.h"

#define COREMP135_I2C_DEVFILE "/dev/i2c-0"
#define COREMP135_UPS_I2C_ADDR (0x34)

static int ups_fd = -1;

static int battery_map(int x, int in_min, int in_max, int out_min, int out_max) {
    float ratio  = (float)(x - in_min) / (in_max - in_min);
    float result = ratio * (out_max - out_min) + out_min;
    return (int)round(result);
}

int ups_init() {
    ups_fd = open(COREMP135_I2C_DEVFILE, O_RDWR);
    if (ups_fd < 0) {
        perror("Failed to open I2C device");
        return -1;
    }
    if (ioctl(ups_fd, I2C_SLAVE_FORCE, COREMP135_UPS_I2C_ADDR) < 0) {
        perror("Failed to set I2C slave address");
        close(ups_fd);
        ups_fd = -1;
        return -1;
    }

    // 初始化配置：写寄存器 0x30 = 0xFF，0x50 = 0x1E
    char buf[2];
    buf[0] = 0x30;
    buf[1] = 0xFF;
    if (write(ups_fd, buf, 2) != 2) {
        perror("Failed to write init reg 0x30");
        return -1;
    }

    buf[0] = 0x50;
    buf[1] = 0x1E;
    if (write(ups_fd, buf, 2) != 2) {
        perror("Failed to write init reg 0x50");
        return -1;
    }

    return 0;
}

// 读取电压寄存器(mV)
int32_t read_ups_voltage() {
    if (ups_fd < 0) {
        return -1;
    }

    char reg = 0x34;
    if (write(ups_fd, &reg, 1) != 1) {
        perror("Failed to set voltage register");
        return -1;
    }

    uint8_t data[2];
    if (read(ups_fd, data, 2) != 2) {
        perror("Failed to read voltage data");
        return -1;
    }

    int32_t vbat = (((data[0] & 0x3f) << 8) | data[1]);
    return vbat;
}

// 读取电池容量寄存器
int32_t read_ups_soc() {
    int32_t vbat = read_ups_voltage();
    if (vbat < 0) {
        return -1;
    }

    // 参考代码将 vbat/10 后映射到 [370,420] -> [0,100]
    // 即 vbat 范围 [3700, 4200] mV 映射到 [0, 100]%
    int tmp_mk = (int)round(vbat / 10.0f);
    if (tmp_mk > 420) tmp_mk = 420;
    if (tmp_mk < 370) tmp_mk = 370;

    return battery_map(tmp_mk, 370, 420, 0, 100);
}
