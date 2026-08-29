#ifndef __NANO_UI_ICON_H__
#define __NANO_UI_ICON_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "graphics.h"

// 图标绘制（带 PSRAM 缓存）：
//   首次请求某路径时，从SD卡读取文件并解码为 RGBA 像素，常驻缓存于 PSRAM；
//   之后再次请求同一路径时，直接用缓存像素混合绘制，省去 SD 读取与 PNG 解码。
// 图标中心点对齐到 (cx, cy)。读取/解码失败时不绘制（并缓存失败结果，避免重复访问SD卡）。
// 返回值：0=已绘制；-1=未绘制（读取/解码失败或缓存已满），调用方可据此回退到替代绘制。
// 注意：缓存永不失效，SD卡上更换图标文件后需复位重建。
int32_t ui_icon_draw_centered(Nano_GFX *gfx, const char *path, int32_t cx, int32_t cy);

// 查询图标实际宽高（像素）：与绘制共用同一缓存，首次查询即触发读取/解码。
// 返回值：0=成功（宽高写入 out_width/out_height，均可传NULL）；-1=读取/解码失败或缓存已满。
int32_t ui_icon_get_size(const char *path, int32_t *out_width, int32_t *out_height);

// 图标旋转绘制（带 PSRAM 缓存，逆映射最近邻采样 + alpha 混合）：
//   贴图上缘横向中点固定于枢轴 (pivot_x, pivot_y)，绕枢轴旋转 angle_rad 弧度
//  （正值向右倾；旋转后贴图局部竖直轴指向 (sinθ, cosθ)，可与绳/摆杆方向对齐）。
// 返回值：0=已绘制；-1=未绘制（读取/解码失败或缓存已满），调用方可据此回退到替代绘制。
int32_t ui_icon_draw_rotated(Nano_GFX *gfx, const char *path, int32_t pivot_x, int32_t pivot_y, float angle_rad);

#ifdef __cplusplus
}
#endif

#endif
