#include "keyboard_hal.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

#include "platform.h"


static int kb_fd;

int keyboard_hal_init() {
    kb_fd = open(KB_I2C_DEVFILE, O_RDWR);
    if(kb_fd < 0) {
        return -1;
    }
    if(ioctl(kb_fd, I2C_SLAVE, KB_I2C_ADDR) < 0) {
        return -1;
    }
    return 0;
}

char keyboard_hal_read_key() {
    char val = 0x03;
    char read_buf = 0;
    if(write(kb_fd, &val, 1) < 0) {
        return -1;
    }
    usleep(1000);
    if(read(kb_fd, &read_buf, 1)) {
        return read_buf;
    }
    else {
        return -1;
    }
}
