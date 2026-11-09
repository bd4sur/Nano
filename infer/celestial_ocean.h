#ifndef __NANO_CELESTIAL_OCEAN_H__
#define __NANO_CELESTIAL_OCEAN_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "graphics.h"

// ===============================================================================
// 实时海洋 + 海洋天空（web-ocean 移植，见 linglong.html === [WebOcean] ===）
//
// 两个互相独立、共享同一套 LUT/波场的手册件：
//   · 海洋天空（散射模型 5）：Hillaire 式 LUT 大气——透射率 LUT + 多重散射 LUT +
//     天空视图 LUT + AgX 色调映射。作为 render_sky 的 sky_model==5 使用。
//   · 实时海洋（地景 3）：JONSWAP 风浪谱 + 窄涌浪的解析波列级数（方差守恒，烘焙为
//     两级可平铺波场纹理并按群速度平流）、LEAN 坡度统计 + 各向异性高斯太阳耀光
//     （Smith 掩蔽）、天空环境反射、水体上涌 + 波峰透射辉光、雅可比白帽、大气透视。
//     作为 render_sky 的 landscape_index==3 使用（只绘制地平线以下的像素）。
//
// 与原网页实现的差异（仅为落地在软件光栅 + 固定观测点上）：
//   · 无岛屿/机场地景：海底高程恒为深水（对应原实现 isl_on=0、bathy=-70m），
//     浅水色与拍岸泡沫因此不出现；
//   · 不做 CPU/GLSL 双路一致性所需的 f16 量化与 LUT 上传，纹理直接留 float32；
//   · 观察者水平位置固定为世界原点（相机不随飞行移动），波场仍按群速度平流。
// ===============================================================================

// 海况 / 大气可调参数（默认值 = linglong.html 中 web-ocean 移植的默认值）
typedef struct Linglong_Ocean_Config {
    float cam_y;        // 观察者高度（平均海平面以上，m）
    float wind_u;       // 10m 风速 m/s
    float wind_dir;     // 波浪行进罗盘方位角（0=北，90=东）
    float fetch;        // 风区长度 m
    float swell_hs;     // 涌浪有效波高 m
    float swell_dir;    // 涌浪方向（罗盘方位角）
    float swell_len;    // 涌浪谱峰波长 m
    float foam;         // 泡沫（白帽）强度，0 = 不显示
    float turbidity;    // 水体浑浊度（0.2 深水暗蓝 ~ 3 浑浊）
    float haze;         // 气溶胶霾倍率（0 = 清透空气）
    float fog_floor;    // 大气透视消光系数 1/m（默认 2e-5 ≈ 150km 能见度）
} Linglong_Ocean_Config;

// 取默认参数（与参考网页一致）
void linglong_ocean_config_default(Linglong_Ocean_Config *cfg);

// 应用参数：任一字段变化都会使 LUT / 波场在下一帧重建（当前实现为整体重建）。
void linglong_ocean_configure(const Linglong_Ocean_Config *cfg);

// 每帧准备：推进海面动画时间、按需重建透射率/多重散射/天空视图 LUT、烘焙波场。
// render_sky 在绘制天空之前调用一次（sky_model==5 或 landscape_index==3 时）。
//   sun_alt / sun_azi：太阳的地平坐标（度；方位角自北向东）
//   view_f / sky_radius：天象仪当前投影参数（用于每像素角尺寸）
//   projection：0-鱼眼 1-线性透视
void linglong_ocean_prepare(float sun_alt, float sun_azi, float view_f, float sky_radius, int32_t projection);

// 海洋天空（散射模型 5）：页面地平系（z 轴向上）射线 → AgX 色调映射后的 RGB（0..1）。
void linglong_ocean_sky_pixel(float ray_x, float ray_y, float ray_z, float *red, float *green, float *blue);

// 实时海洋地景：射线步进求交 + 物理着色，只写地平线以下的像素
// （调用方负责在此之前已完成天空/恒星等绘制；本函数覆盖地平线以下的画面）。
void linglong_ocean_draw_landscape(
    Nano_GFX *gfx,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f, int32_t projection);

// 释放全部堆资源（LUT / 波场 / 泡沫纹理）
void linglong_ocean_free(void);

#ifdef __cplusplus
}
#endif

#endif
