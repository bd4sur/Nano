#ifndef __NANO_UI_GOLDMINER_H__
#define __NANO_UI_GOLDMINER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 黄金矿工
//
// 玩法：
//   - 钩子在矿工下方左右摆动，硬按键 D(回车)/2 或点击屏幕任意位置沿当前方向发射；
//   - 钩尖触到金块/岩石/钻石/福袋/宝箱即抓住并回收，回收速度受物体重量影响；
//   - 物品拉到顶部后计入得分；清空全部物品后进入下一关（更多物品、摆动更快）；
//   - 硬按键 A(ESC) 或点击右上角“返回”虚拟按钮返回小游戏菜单（软按键事件不响应）。
//
// 贴图预留：全部实体经 gm_draw_sprite 按精灵ID绘制（见 ui_goldminer.c 头部说明），
//   当前用基本图形（圆/矩形/三角形）绘制原型；后续为精灵表配置贴图路径即可切换为贴图。
// ===============================================================================

int32_t ui_goldminer_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_goldminer_render_frame(Key_Event *key_event, Global_State *global_state);
int32_t ui_goldminer_event_handler(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
