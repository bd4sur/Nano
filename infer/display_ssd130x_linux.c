#include "display_hal.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <wchar.h>

#define OLED_CMD 0  // 写命令
#define OLED_DATA 1 // 写数据

#define OLED_Command_Mode 0x00
#define OLED_Data_Mode    0x40

static int i2cdev_fd;

// 发送命令
void OLED_WriteData(uint8_t data) {
    uint8_t buf[2];
    buf[0] = OLED_Data_Mode;
    buf[1] = data;
    if( write(i2cdev_fd, &buf, 2) != 2 ) printf("I2C write error\n");
    /*
    Wire.beginTransmission(OLED_I2C_ADDR);
    Wire.write(OLED_Data_Mode);
    Wire.write(data);
    Wire.endTransmission();
    */
}

// 发送数据
void OLED_WriteCommand(uint8_t cmd) {
    uint8_t buf[2];
    buf[0] = OLED_Command_Mode;
    buf[1] = cmd;
    if( write(i2cdev_fd, &buf, 2) != 2 ) printf("I2C write error\n");
    /*
    Wire.beginTransmission(OLED_I2C_ADDR);
    Wire.write(OLED_Command_Mode);
    Wire.write(cmd);
    Wire.endTransmission();
    */
}

// 更新显存到OLED
void display_hal_refresh(uint8_t **FRAME_BUFFER) {
    for (uint8_t row = 0; row < FB_PAGES; row++) {
        uint8_t col = 0;
        OLED_WriteCommand(0xb0 + row); // 设置行起始地址
        OLED_WriteCommand(0x00 + (col & 0x0F));        // 设置低列起始地址
        OLED_WriteCommand(0x10 + (col & 0x0F)); // 设置高列起始地址

        // for (uint8_t col = 0; col < FB_WIDTH; col++) {
        //     OLED_WriteData(FRAME_BUFFER[row][col]);
        // }

        uint8_t buf[129];
        buf[0] = OLED_Data_Mode;
        for (uint8_t col = 0; col < FB_WIDTH; col++) {
            buf[col + 1] = FRAME_BUFFER[row][col];
        }
        if( write(i2cdev_fd, buf, (FB_WIDTH+1)) != (FB_WIDTH+1) ) printf("I2C write error\n");

        // Wire.beginTransmission(OLED_I2C_ADDR);
        // Wire.write(OLED_Data_Mode);
        // Wire.write(FRAME_BUFFER[row], 128);
        // Wire.endTransmission();
    
        // 有32字节的缓冲区??
/*
        while (col < FB_WIDTH) {
            uint8_t buf[64];
            buf[0] = OLED_Data_Mode;
            uint8_t actual_buf_length = 1;
            for (uint8_t i = 1; i < 64; i++) {
                buf[i] = FRAME_BUFFER[row][col];
                col++;
                actual_buf_length++;
                if(col >= FB_WIDTH) break;
            }
            if( write(i2cdev_fd, buf, actual_buf_length) != actual_buf_length ) printf("I2C write error\n");
        }
*/
    }
}

// OLED的初始化
void display_hal_init(void) {
    // 初始化屏幕设备
    i2cdev_fd = open(OLED_I2C_DEVFILE, O_RDWR);
    if (i2cdev_fd < 0) {
        printf("OLED open error : %s\r\n", strerror(errno));
    }
    if (ioctl(i2cdev_fd, I2C_SLAVE, OLED_I2C_ADDR) < 0) {
        printf("OLED ioctl error : %s\r\n", strerror(errno));
    }

    usleep(100*1000);

#ifdef SSD1309

    // SSD1309 ///////////////////////////////////////////////////////////////////////////////

    OLED_WriteCommand(0xFD);
    OLED_WriteCommand(0x12);
    OLED_WriteCommand(0xAE); //--turn off oled panel
    OLED_WriteCommand(0xd5); //--set display clock divide ratio/oscillator frequency
    OLED_WriteCommand(0xf0);
    OLED_WriteCommand(0xA8); //--set multiplex ratio(1 to 64)
    OLED_WriteCommand(0x3f); //--1/64 duty
    OLED_WriteCommand(0xD3); //-set display offset  Shift Mapping RAM Counter (0x00~0x3F)
    OLED_WriteCommand(0x00); //-not offset
    OLED_WriteCommand(0x40); //--set start line address  Set Mapping RAM Display Start Line (0x00~0x3F)
    OLED_WriteCommand(0xA1); //--Set SEG/Column Mapping     0xa0左右反置 0xa1正常
    OLED_WriteCommand(0xC8); // Set COM/Row Scan Direction   0xc0上下反置 0xc8正常
    OLED_WriteCommand(0xDA); //--set com pins hardware configuration
    OLED_WriteCommand(0x12);
    OLED_WriteCommand(0x81); //--set contrast control register
    OLED_WriteCommand(0xFF); // Set SEG Output Current Brightness
    OLED_WriteCommand(0xD9); //--set pre-charge period
    OLED_WriteCommand(0x82); // Set Pre-Charge as 15 Clocks & Discharge as 1 Clock
    OLED_WriteCommand(0xDB); //--set vcomh
    OLED_WriteCommand(0x34); // Set VCOM Deselect Level
    OLED_WriteCommand(0xA4); // Disable Entire Display On (0xa4/0xa5)
    OLED_WriteCommand(0xA6); // Disable Inverse Display On (0xa6/a7)
    OLED_WriteCommand(0x2E); // Stop scroll
    OLED_WriteCommand(0x20); // Set Memory Addressing Mode
    OLED_WriteCommand(0x00); // Set Memory Addressing Mode ab Horizontal addressing mode
    OLED_WriteCommand(0xAF);

#elif defined(SSD1306)

    // SSD1306 ///////////////////////////////////////////////////////////////////////////////

    OLED_WriteCommand(0xAE); // Display OFF
    OLED_WriteCommand(0xD5); // Set Display Clock Divide Ratio / Oscillator Frequency
    OLED_WriteCommand(0x80); // SSD1306 推荐值：0x80 (divide ratio = 1, freq = 8)
    OLED_WriteCommand(0xA8); // Set Multiplex Ratio
    OLED_WriteCommand(0x3F); // 1/64 duty (for 64 height)
    OLED_WriteCommand(0xD3); // Set Display Offset
    OLED_WriteCommand(0x00); // No offset
    OLED_WriteCommand(0x40); // Set Start Line (0x40 = 0)
    OLED_WriteCommand(0x8D); // Charge Pump Setting
    OLED_WriteCommand(0x14); // Enable charge pump (REQUIRED for SSD1306!)
    OLED_WriteCommand(0x20); // Memory Addressing Mode
    OLED_WriteCommand(0x00); // Horizontal Addressing Mode
    OLED_WriteCommand(0xA1); // Segment Re-map: column 127 mapped to SEG0 (正常左右方向)
    OLED_WriteCommand(0xC8); // COM Output Scan Direction: remapped (正常上下方向)
    OLED_WriteCommand(0xDA); // Set COM Pins Hardware Configuration
    OLED_WriteCommand(0x12); // Alternative COM pin configuration, disable COM left/right remap
    OLED_WriteCommand(0x81); // Set Contrast Control
    OLED_WriteCommand(0xCF); // SSD1306 typical value (0x7F~0xFF, 0xCF is bright)
    OLED_WriteCommand(0xD9); // Set Pre-Charge Period
    OLED_WriteCommand(0xF1); // SSD1306 typical: phase1 = 15, phase2 = 1 (0xF1)
    OLED_WriteCommand(0xDB); // Set VCOMH Deselect Level
    OLED_WriteCommand(0x40); // SSD1306 recommended (0x20, 0x30, 0x40 are common)
    OLED_WriteCommand(0xA4); // Entire Display ON (resume to RAM content)
    OLED_WriteCommand(0xA6); // Normal Display (not inverse)
    OLED_WriteCommand(0xAF); // Display ON
    // Page mode
    OLED_WriteCommand(0x20);
    OLED_WriteCommand(0x02);

#endif

}

void display_hal_close(void) {
    close(i2cdev_fd);
}

// 开启OLED显示
void OLED_Display_On(void) {
    OLED_WriteCommand(0x8D); // 电荷泵使能
    OLED_WriteCommand(0x14); // 开启电荷泵
    OLED_WriteCommand(0xAF); // 点亮屏幕
}

// 关闭OLED显示
void OLED_Display_Off(void) {
    OLED_WriteCommand(0x8D); // 电荷泵使能
    OLED_WriteCommand(0x10); // 关闭电荷泵
    OLED_WriteCommand(0xAE); // 关闭屏幕
}

// 反显函数
void OLED_ColorTurn(uint8_t i) {
    if (i == 0) {
        OLED_WriteCommand(0xA6); // 正常显示
    }
    if (i == 1) {
        OLED_WriteCommand(0xA7); // 反色显示
    }
}

// 屏幕旋转180度
void OLED_DisplayTurn(uint8_t i) {
    if (i == 0) {
        OLED_WriteCommand(0xC8); // 正常显示
        OLED_WriteCommand(0xA1);
    }
    if (i == 1) {
        OLED_WriteCommand(0xC0); // 反转显示
        OLED_WriteCommand(0xA0);
    }
}
