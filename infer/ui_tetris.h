#ifndef __NANO_UI_TETRIS_H__
#define __NANO_UI_TETRIS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 俄罗斯方块
//
// 操作（十六宫格按键）：
//   ←/→ 左右移动   1-旋转   2-加速下落   D(回车)-直接落底   A(ESC)-返回小游戏菜单
// 规则：10x20 标准场地，7 种方块，消行计分（100/300/500/800 × 关卡），
//   每消 10 行升一关，下落速度随关卡加快；新方块无法入场即游戏结束。
// ===============================================================================

int32_t ui_tetris_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_tetris_render_frame(Key_Event *key_event, Global_State *global_state);
int32_t ui_tetris_event_handler(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
