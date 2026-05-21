#ifndef __NANO_KEYBOARD_H__
#define __NANO_KEYBOARD_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"

// 逻辑键值定义
// ITU E.161 12键电话键盘+4个额外案件=16按键
#define KEYCODE_NUM_0 (0)
#define KEYCODE_NUM_1 (1)
#define KEYCODE_NUM_2 (2)
#define KEYCODE_NUM_3 (3)
#define KEYCODE_NUM_4 (4)
#define KEYCODE_NUM_5 (5)
#define KEYCODE_NUM_6 (6)
#define KEYCODE_NUM_7 (7)
#define KEYCODE_NUM_8 (8)
#define KEYCODE_NUM_9 (9)
#define KEYCODE_NUM_A (10)
#define KEYCODE_NUM_B (11)
#define KEYCODE_NUM_C (12)
#define KEYCODE_NUM_D (13)
#define KEYCODE_NUM_STAR (14)
#define KEYCODE_NUM_HASH (15)
#define KEYCODE_NUM_IDLE (127)

/******************************************************

屏幕区域布局示意
    +---------+---------+---------+---------+
    |              Padding Top              |
+---0---------1---------2---------3---------4---+
|   | (0,0) 1 | (1,0) 2 | (2,0) 3 | (3,0) A |   |
| L 1---------+---------+---------+---------+ R |
| E | (0,1) 4 | (1,1) 5 | (2,1) 6 | (3,1) B | I |
| F 2---------+---------+---------+---------+ G |
| T | (0,2) 7 | (1,2) 8 | (2,2) 9 | (3,2) C | H |
|   3---------+---------+---------+---------+ T |
|   | (0,3) E*| (1,3) 0 | (2,3) F#| (3,3) D |   |
+---4---------+---------+---------+---------+---|
    |            Padding Bottom             |
    +---------+---------+---------+---------+

******************************************************/
// 屏幕布局坐标
#define PADDING_TOP    (14)
#define PADDING_BOTTOM (14)
#define PADDING_LEFT   (0)
#define PADDING_RIGHT  (0)
#define CELL_NUM_X     (4)
#define CELL_NUM_Y     (4)
#define CELL_WIDTH     ((SCREEN_WIDTH-PADDING_LEFT-PADDING_RIGHT)/(CELL_NUM_X))
#define CELL_HEIGHT    ((SCREEN_HEIGHT-PADDING_TOP-PADDING_BOTTOM)/(CELL_NUM_Y))
#define CELL_X0(col,row)       (PADDING_LEFT + (col) * CELL_WIDTH)
#define CELL_Y0(col,row)       (PADDING_TOP  + (row) * CELL_HEIGHT)
#define CELL_CENTER_X(col,row) (CELL_X0((col),(row)) + (CELL_WIDTH/2))
#define CELL_CENTER_Y(col,row) (CELL_Y0((col),(row)) + (CELL_HEIGHT/2))

// 网格范围
#define IN_BTN_1(x,y)  (((x) >= CELL_X0(0,0)) && ((x) < CELL_X0(1,0)) && ((y) >= CELL_Y0(0,0)) && ((y) < CELL_Y0(0,1)))
#define IN_BTN_2(x,y)  (((x) >= CELL_X0(1,0)) && ((x) < CELL_X0(2,0)) && ((y) >= CELL_Y0(1,0)) && ((y) < CELL_Y0(1,1)))
#define IN_BTN_3(x,y)  (((x) >= CELL_X0(2,0)) && ((x) < CELL_X0(3,0)) && ((y) >= CELL_Y0(2,0)) && ((y) < CELL_Y0(2,1)))
#define IN_BTN_A(x,y)  (((x) >= CELL_X0(3,0)) && ((x) < CELL_X0(4,0)) && ((y) >= CELL_Y0(3,0)) && ((y) < CELL_Y0(3,1)))
#define IN_BTN_4(x,y)  (((x) >= CELL_X0(0,1)) && ((x) < CELL_X0(1,1)) && ((y) >= CELL_Y0(0,1)) && ((y) < CELL_Y0(0,2)))
#define IN_BTN_5(x,y)  (((x) >= CELL_X0(1,1)) && ((x) < CELL_X0(2,1)) && ((y) >= CELL_Y0(1,1)) && ((y) < CELL_Y0(1,2)))
#define IN_BTN_6(x,y)  (((x) >= CELL_X0(2,1)) && ((x) < CELL_X0(3,1)) && ((y) >= CELL_Y0(2,1)) && ((y) < CELL_Y0(2,2)))
#define IN_BTN_B(x,y)  (((x) >= CELL_X0(3,1)) && ((x) < CELL_X0(4,1)) && ((y) >= CELL_Y0(3,1)) && ((y) < CELL_Y0(3,2)))
#define IN_BTN_7(x,y)  (((x) >= CELL_X0(0,2)) && ((x) < CELL_X0(1,2)) && ((y) >= CELL_Y0(0,2)) && ((y) < CELL_Y0(0,3)))
#define IN_BTN_8(x,y)  (((x) >= CELL_X0(1,2)) && ((x) < CELL_X0(2,2)) && ((y) >= CELL_Y0(1,2)) && ((y) < CELL_Y0(1,3)))
#define IN_BTN_9(x,y)  (((x) >= CELL_X0(2,2)) && ((x) < CELL_X0(3,2)) && ((y) >= CELL_Y0(2,2)) && ((y) < CELL_Y0(2,3)))
#define IN_BTN_C(x,y)  (((x) >= CELL_X0(3,2)) && ((x) < CELL_X0(4,2)) && ((y) >= CELL_Y0(3,2)) && ((y) < CELL_Y0(3,3)))
#define IN_BTN_E(x,y)  (((x) >= CELL_X0(0,3)) && ((x) < CELL_X0(1,3)) && ((y) >= CELL_Y0(0,3)) && ((y) < CELL_Y0(0,4)))
#define IN_BTN_0(x,y)  (((x) >= CELL_X0(1,3)) && ((x) < CELL_X0(2,3)) && ((y) >= CELL_Y0(1,3)) && ((y) < CELL_Y0(1,4)))
#define IN_BTN_F(x,y)  (((x) >= CELL_X0(2,3)) && ((x) < CELL_X0(3,3)) && ((y) >= CELL_Y0(2,3)) && ((y) < CELL_Y0(2,4)))
#define IN_BTN_D(x,y)  (((x) >= CELL_X0(3,3)) && ((x) < CELL_X0(4,3)) && ((y) >= CELL_Y0(3,3)) && ((y) < CELL_Y0(3,4)))

#define IN_BTN_TOP(x,y)     ((y) < CELL_Y0(0,0))
#define IN_BTN_BOTTOM(x,y)  ((y) >= CELL_Y0(0,4))


int32_t input_device_init();
uint8_t input_device_read_key();

#ifdef __cplusplus
}
#endif

#endif