#ifndef __NANO_UI_FLIP_H__
#define __NANO_UI_FLIP_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 番茄表 / 电子沙漏（STATE_FLIP 的状态页）
//
// 界面：顶栏（状态灯 + 标题 + 状态胶囊 + 模式胶囊 + 数码计时 + 退出）、中央 4:3 流体舞台、
// 左右仪表（进度 / 节流度）、底部控制条（复位 / 画风 / 节流开关 / 配色 / 流量读数），
// 外加模态配色选择器。全部触屏热点自绘、松手沿结算；实体键盘仅作兜底。
//
// 两种模式：
//   · 沙漏（默认）：flip.c 的沙漏实体 + 节流，流完冻结计时并振动报警；
//   · 自由流体：同一求解器换成矩形外壁（hourglass_enabled=0）、无流完判定、计时退化为秒表。
//
// 流体求解与渲染在 flip.c（flip_init / render_flip / flip_set_color_style），本模块只管
// 界面与交互；320x240 为设计基准，其余分辨率按画布宽高等比缩放。
// ===============================================================================

// 流体网格分辨率（flip_init 的 resolution 参数，影响密度网格的格子数）
#define FLIP_RESOLUTION (24)

void ui_flip_init(Key_Event *key_event, Global_State *global_state);
void ui_flip_render_frame(Key_Event *key_event, Global_State *global_state);
void ui_flip_event_handler(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
