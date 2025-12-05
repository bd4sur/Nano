#ifndef __OLED_H__
#define __OLED_H__


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

#include "platform.h"
#include "utils.h"

#define STRING_BUFFER_LENGTH (65536)

#define OLED_WIDTH  128
#define OLED_HEIGHT 64
#define OLED_PAGES  8

#define OLED_CMD 0  // 写命令
#define OLED_DATA 1 // 写数据

#define OLED_Command_Mode 0x00
#define OLED_Data_Mode    0x40

#define u8 unsigned char
#define u16 unsigned int
#define u32 unsigned long

#define delayMicroseconds(x) usleep(x)
#define delay(x) usleep((x)*1000)

#define MIN(x, y) (((x) > (y)) ? (y) : (x))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

void OLED_WriteByte(uint8_t data, uint8_t mode);
void OLED_Refresh(uint8_t **FRAME_BUFFER);
void OLED_Init(void);
void OLED_Close(void);
void OLED_Display_On(void);
void OLED_Display_Off(void);
void OLED_ColorTurn(uint8_t i);
void OLED_DisplayTurn(uint8_t i);



#endif
