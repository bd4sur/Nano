#include "input_device.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

#include "platform.h"

// 物理键值定义
// ITU E.161 12键电话键盘+4个额外按键=16按键
#define KEYCODE_NUM_0    (0)
#define KEYCODE_NUM_1    (1)
#define KEYCODE_NUM_2    (2)
#define KEYCODE_NUM_3    (3)
#define KEYCODE_NUM_4    (4)
#define KEYCODE_NUM_5    (5)
#define KEYCODE_NUM_6    (6)
#define KEYCODE_NUM_7    (7)
#define KEYCODE_NUM_8    (8)
#define KEYCODE_NUM_9    (9)
#define KEYCODE_NUM_A    (10)
#define KEYCODE_NUM_B    (11)
#define KEYCODE_NUM_C    (12)
#define KEYCODE_NUM_D    (13)
#define KEYCODE_NUM_STAR (14)
#define KEYCODE_NUM_HASH (15)
#define KEYCODE_NUM_IDLE (16)

static int kb_fd;

int32_t input_device_init() {
    kb_fd = open(I2C_DEVFILE, O_RDWR);
    if(kb_fd < 0) {
        return -1;
    }
    if(ioctl(kb_fd, I2C_SLAVE, KB_I2C_ADDR) < 0) {
        return -1;
    }
    return 0;
}

uint8_t input_device_read_key() {
    uint8_t val = 0x03;
    uint8_t read_buf = 0;
    if(write(kb_fd, &val, 1) < 0) {
        return NANO_KEY_IDLE;
    }
    usleep(1000);
    if(read(kb_fd, &read_buf, 1)) {
        switch(read_buf) {
            case KEYCODE_NUM_0:    return NANO_KEY_0;
            case KEYCODE_NUM_1:    return NANO_KEY_1;
            case KEYCODE_NUM_2:    return NANO_KEY_2;
            case KEYCODE_NUM_3:    return NANO_KEY_3;
            case KEYCODE_NUM_4:    return NANO_KEY_4;
            case KEYCODE_NUM_5:    return NANO_KEY_5;
            case KEYCODE_NUM_6:    return NANO_KEY_6;
            case KEYCODE_NUM_7:    return NANO_KEY_7;
            case KEYCODE_NUM_8:    return NANO_KEY_8;
            case KEYCODE_NUM_9:    return NANO_KEY_9;
            case KEYCODE_NUM_A:    return NANO_KEY_esc;
            case KEYCODE_NUM_B:    return NANO_KEY_shift;
            case KEYCODE_NUM_C:    return NANO_KEY_ctrl;
            case KEYCODE_NUM_D:    return NANO_KEY_enter;
            case KEYCODE_NUM_STAR: return NANO_KEY_left;
            case KEYCODE_NUM_HASH: return NANO_KEY_right;
            case KEYCODE_NUM_IDLE: return NANO_KEY_IDLE;

            default: return NANO_KEY_IDLE;
        }
    }
    else {
        return NANO_KEY_IDLE;
    }
}
