#ifndef __NANO_KEYBOARD_H__
#define __NANO_KEYBOARD_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"


// 逻辑键值定义
// ASCII: !"#$%&'()*+,-./0123456789:;<=>?​@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_​`abcdefghijklmnopqrstuvwxyz{|}~
//       ^32d

#define NANO_KEY_IDLE (0)

#define NANO_KEY_up (1)
#define NANO_KEY_down (2)
#define NANO_KEY_left (3)
#define NANO_KEY_right (4)
#define NANO_KEY_ctrl (5)
#define NANO_KEY_alt (6)
#define NANO_KEY_shift (7)
#define NANO_KEY_backspace (8)
#define NANO_KEY_tab (9)
#define NANO_KEY_capslock (10)
#define NANO_KEY_super (11)
#define NANO_KEY_menu (12)
#define NANO_KEY_enter (13)
#define NANO_KEY_fn (14)
#define NANO_KEY_f1 (15)
#define NANO_KEY_f2 (16)
#define NANO_KEY_f3 (17)
#define NANO_KEY_f4 (18)
#define NANO_KEY_f5 (19)
#define NANO_KEY_f6 (20)
#define NANO_KEY_f7 (21)
#define NANO_KEY_f8 (22)
#define NANO_KEY_f9 (23)
#define NANO_KEY_f10 (24)
#define NANO_KEY_f11 (25)
#define NANO_KEY_f12 (26)
#define NANO_KEY_esc (27)
#define NANO_KEY_home (28)
#define NANO_KEY_end (29)
#define NANO_KEY_pgup (30)
#define NANO_KEY_pgdn (31)

#define NANO_KEY_space (32)
#define NANO_KEY_bang (33)
#define NANO_KEY_quote2 (34)
#define NANO_KEY_hash (35)
#define NANO_KEY_dollar (36)
#define NANO_KEY_percent (37)
#define NANO_KEY_and (38)
#define NANO_KEY_quote1 (39)
#define NANO_KEY_parenl (40)
#define NANO_KEY_parenr (41)
#define NANO_KEY_star (42)
#define NANO_KEY_plus (43)
#define NANO_KEY_comma (44)
#define NANO_KEY_dash (45)
#define NANO_KEY_dot (46)
#define NANO_KEY_slash (47)
#define NANO_KEY_0 (48)
#define NANO_KEY_1 (49)
#define NANO_KEY_2 (50)
#define NANO_KEY_3 (51)
#define NANO_KEY_4 (52)
#define NANO_KEY_5 (53)
#define NANO_KEY_6 (54)
#define NANO_KEY_7 (55)
#define NANO_KEY_8 (56)
#define NANO_KEY_9 (57)
#define NANO_KEY_colon (58)
#define NANO_KEY_semicolon (59)
#define NANO_KEY_lt (60)
#define NANO_KEY_eq (61)
#define NANO_KEY_gt (62)
#define NANO_KEY_ques (63)
#define NANO_KEY_at (64)
#define NANO_KEY_A (65)
#define NANO_KEY_B (66)
#define NANO_KEY_C (67)
#define NANO_KEY_D (68)
#define NANO_KEY_E (69)
#define NANO_KEY_F (70)
#define NANO_KEY_G (71)
#define NANO_KEY_H (72)
#define NANO_KEY_I (73)
#define NANO_KEY_J (74)
#define NANO_KEY_K (75)
#define NANO_KEY_L (76)
#define NANO_KEY_M (77)
#define NANO_KEY_N (78)
#define NANO_KEY_O (79)
#define NANO_KEY_P (80)
#define NANO_KEY_Q (81)
#define NANO_KEY_R (82)
#define NANO_KEY_S (83)
#define NANO_KEY_T (84)
#define NANO_KEY_U (85)
#define NANO_KEY_V (86)
#define NANO_KEY_W (87)
#define NANO_KEY_X (88)
#define NANO_KEY_Y (89)
#define NANO_KEY_Z (90)
#define NANO_KEY_bracketl (91)
#define NANO_KEY_backslash (92)
#define NANO_KEY_bracketr (93)
#define NANO_KEY_caret (94)
#define NANO_KEY_underscore (95)
#define NANO_KEY_backtick (96)
#define NANO_KEY_a (97)
#define NANO_KEY_b (98)
#define NANO_KEY_c (99)
#define NANO_KEY_d (100)
#define NANO_KEY_e (101)
#define NANO_KEY_f (102)
#define NANO_KEY_g (103)
#define NANO_KEY_h (104)
#define NANO_KEY_i (105)
#define NANO_KEY_j (106)
#define NANO_KEY_k (107)
#define NANO_KEY_l (108)
#define NANO_KEY_m (109)
#define NANO_KEY_n (110)
#define NANO_KEY_o (111)
#define NANO_KEY_p (112)
#define NANO_KEY_q (113)
#define NANO_KEY_r (114)
#define NANO_KEY_s (115)
#define NANO_KEY_t (116)
#define NANO_KEY_u (117)
#define NANO_KEY_v (118)
#define NANO_KEY_w (119)
#define NANO_KEY_x (120)
#define NANO_KEY_y (121)
#define NANO_KEY_z (122)
#define NANO_KEY_bracel (123)
#define NANO_KEY_pipe (124)
#define NANO_KEY_bracer (125)
#define NANO_KEY_tilde (126)

#define NANO_KEY_del (127)




// ASCII: !"#$%&'()*+,-./0123456789:;<=>?​@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_​`abcdefghijklmnopqrstuvwxyz{|}~
//       ^32d

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