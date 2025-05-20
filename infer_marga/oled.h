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

#include "avltree.h"


///////////////////////////////////////////////
#define OLED_I2C_DEVFILE "/dev/i2c-0"
#define OLED_I2C_ADDR 0x3c
///////////////////////////////////////////////

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
void OLED_Init(void);
void OLED_Close(void);
void OLED_Display_On(void);
void OLED_Display_Off(void);
void OLED_Refresh(void);
void OLED_Clear(void);
void OLED_SoftClear(void);
void OLED_DrawPoint(uint8_t x, uint8_t y, uint8_t t);
void OLED_DrawLine(uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, uint8_t mode);
void OLED_DrawCircle(uint8_t x, uint8_t y, uint8_t r);
void OLED_ShowChar(uint8_t x, uint8_t y, uint8_t *glyph, uint8_t font_width, uint8_t font_height, uint8_t mode);
void OLED_ShowPicture(uint8_t x, uint8_t y, uint8_t sizex, uint8_t sizey, uint8_t BMP[], uint8_t mode);

void OLED_ColorTurn(uint8_t i);
void OLED_DisplayTurn(uint8_t i);

uint8_t *get_glyph(uint32_t utf32, uint8_t *font_width, uint8_t *font_height);
void render_line(wchar_t *line, uint32_t x, uint32_t y, uint8_t mode);
int32_t render_text(wchar_t *text, int32_t line_pos);

#endif
