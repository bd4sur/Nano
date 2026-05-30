#include "input_device.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

#include "platform.h"


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
        return KEYCODE_NUM_IDLE;
    }
    usleep(1000);
    if(read(kb_fd, &read_buf, 1)) {
        if (read_buf == 16) read_buf = KEYCODE_NUM_IDLE;
        return read_buf;
    }
    else {
        return KEYCODE_NUM_IDLE;
    }
}
