#ifndef __NANO_UI_WATER_H__
#define __NANO_UI_WATER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 水池（WebGL Water）—— 忠实移植 madebyevan.com/webgl-water（water/ 目录，MIT）
//
// 本模块将原作的 GLSL 渲染管线逐行直译为 C 浮点实现，包括：
//   - 水面（高度场）仿真：updateShader（波动方程）、normalShader（法线场）、
//     sphereShader（圆球对水体积的排开/回填，moveSphere）——仿真是水“体积”
//     的运动（水面是其自由面），与原版一致，不加入原作以外的东西（无落水/雨滴）；
//   - 光学：水面片段着色器逐像素追迹 反射/折射（折射率、菲涅尔、全内反射）、
//     池壁/池底着色（瓷砖 + 环境光遮蔽 + 球影）、水中圆球的着色（壁面/水面环境
//     光遮蔽 + 焦散）、天空采样（本机以程序化渐变天空直接绘制代替天空盒贴图）；
//   - 焦散：把水面网格沿折射光方向投影到池底生成的焦散纹理（原 causticsShader，
//     含光强聚焦亮度、圆球挡光阴影、池沿阴影）；
//   - 相机：perspective + 自身 modelview（translate/rotate 组合），
//     2/4/6/8 键旋转视角（按住连转）；拖动旋转视角保留（宏 WT_ENABLE_DRAG_ROTATE，默认关）；
//   - 圆球物理：重力/水中黏滞浮力/池底反弹（更新频率、速度单位与原版一致）。
//
// 按键（十六宫格触控）：A(右上角)返回、D(右下角)开始/暂停、*(左下角)重力、
// 0(* 右侧相邻键)定向光源、2/4/6/8 旋转视角；触摸拖动圆球移动。
// 瓷砖与天空按需求保留为“直接绘制”（程序化生成），不加载贴图文件。
// 定位即为忠实复刻，性能不被优先考虑（软渲染、无 GPU），帧率低是预期行为。
// ===============================================================================

int32_t ui_water_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_water_event_handler(Key_Event *key_event, Global_State *global_state);
int32_t ui_water_render_frame(Key_Event *key_event, Global_State *global_state);
void    ui_water_on_exit(void);

#ifdef __cplusplus
}
#endif

#endif
