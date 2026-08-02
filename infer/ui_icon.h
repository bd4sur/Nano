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
// 注意：缓存永不失效，SD卡上更换图标文件后需复位重建。
void ui_icon_draw_centered(Nano_GFX *gfx, const char *path, int32_t cx, int32_t cy);

#ifdef __cplusplus
}
#endif

#endif
