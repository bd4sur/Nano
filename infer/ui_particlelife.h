#ifndef __NANO_UI_PARTICLELIFE_H__
#define __NANO_UI_PARTICLELIFE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 粒子生命（Particle Life）
//
// 移植自 particle.html（shapoco.net/particlelife）：6 种颜色的粒子在 2x1.5
// （与机器屏幕一致的 4:3）环面世界中按随机生成的种间吸引/排斥参数相互作用，
// 涌现出类生命的自组织图样。
// 原网页每次刷新生成不同参数；本实现按 D(回车) 重新随机生成一套参数与世界。
//
// 原作的"区块 + 邻居克隆"结构在单区块（所有邻居指向自身）时，等价于周期边界
// 条件下的全对相互作用；本实现据此简化。
//
// 实现包含定点/浮点双版本，由 ui_particlelife.c 顶部 PL_USE_FIXED_POINT 宏切换
//（物理常数统一以浮点形式给出，定点版在运行时换算）；定点版针对 ESP32 无 FPU
// 优化（Q20/Q30/Q40 + 整数 rsqrt 查表 + 无序对对称化 + 均匀网格），浮点版保留
// 作算法可读性参考。
//
// 玩法：
//   - 进入即自动生成一套随机相互作用参数与初始分布；
//   - 按 D(回车) 重新生成（等价于原网页刷新页面）；
//   - 按 A(ESC) 返回小游戏菜单。
// ===============================================================================

int32_t ui_particlelife_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_particlelife_event_handler(Key_Event *key_event, Global_State *global_state);
int32_t ui_particlelife_render_frame(Key_Event *key_event, Global_State *global_state);
void    ui_particlelife_on_exit(void);

#ifdef __cplusplus
}
#endif

#endif
