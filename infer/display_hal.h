#ifndef __NANO_DISPLAY_HAL_H__
#define __NANO_DISPLAY_HAL_H__

#ifdef __cplusplus
extern "C" {
#endif


#include "platform.h"
#include "utils.h"

#define FB_WIDTH  128
#define FB_HEIGHT 64
#define FB_PAGES  8

void display_hal_refresh(uint8_t **FRAME_BUFFER);
void display_hal_init(void);
void display_hal_close(void);


// void OLED_Display_On(void);
// void OLED_Display_Off(void);
// void OLED_ColorTurn(uint8_t i);
// void OLED_DisplayTurn(uint8_t i);

#ifdef __cplusplus
}
#endif

#endif
