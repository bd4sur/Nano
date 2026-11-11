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
//     作为 render_sky 的 landscape_index==3 使用。
//   · 随机岛屿地形（见文件末的 Linglong_Terrain_Config）：可平铺周期高程图 + 两级
//     “格内最大高程”网格的射线求交、地形着色、浅水色与拍岸泡沫；地形可露在地平线以上。
//
// 与原网页实现的差异（仅为落地在软件光栅 + 本项目的相机模型上）：
//   · 不做 CPU/GLSL 双路一致性所需的 f16 量化与 LUT 上传，纹理直接留 float32；
//   · 观察者位置（x/y/z）由 render_sky 每帧给出，可水平/垂直自由移动：
//     水平按世界周期回绕、高度下限 1.5 m 并按地形做“不钻地”钳制；
//     波场/泡沫/海底高程都锚定在世界坐标上，因此飞行时浪会从身边流过。
// ===============================================================================

// 海况 / 大气可调参数（默认值 = linglong.html 中 web-ocean 移植的默认值，
// 其中观察高度按需求取 5m、泡沫 0、浑浊度取区间最小值、霾 0、能见度取最大值）
typedef struct Linglong_Ocean_Config {
    float cam_y;        // 观察者高度（m）；由 linglong_ocean_set_observer 同步，勿直接设
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

// ===============================================================================
// 可调档位表（玲珑仪设置界面用；档位序号存在 Linglong_Config，本模块负责映射到物理量）
//   level   ：档位序号（0 起；Linglong_Config 里存的就是它）
//   name    ：设置按钮上显示的现值文字
//   value   ：该档对应的物理量（m）
// ===============================================================================
typedef struct {
    int32_t        level;
    const wchar_t *name;
    float          value;
} Linglong_Ocean_Option;

// 观察高度：5 / 50 / 200 / 1000 / 20000 m（默认 5m）
// 涌浪波高：无 / 0.5 / 1.0 / 1.8（默认）/ 2.5 / 3.5 m
// 涌浪波长：30 / 60 / 90 / 120（默认）/ 160 / 220 / 300 m
const Linglong_Ocean_Option *linglong_ocean_height_options(int32_t *count);
const Linglong_Ocean_Option *linglong_ocean_swell_hs_options(int32_t *count);
const Linglong_Ocean_Option *linglong_ocean_swell_len_options(int32_t *count);

// 按档位序号更新海况（其余参数保持模块默认：风速 7m/s@118°、风区 90km、
// 涌浪方向 94°、泡沫 0、浑浊度 0.2、霾 0、能见度 150km）。
// 内部只重建真正受影响的部分（涌浪 → 只重烘波场），因此切档不会付出全部烘焙代价。
// 观察高度不再经由本函数：观察者位置由 linglong_ocean_set_observer 每帧给出
// （漫游模式下高度是连续的，档位表只作为界面上的快捷跳转值）。
void linglong_ocean_set_levels(int32_t swell_hs_level, int32_t swell_len_level);

// ===============================================================================
// 观察者（相机）世界位置
//
// 坐标系与海洋一致：x 东、y 上（平均海平面 0）、z 北，单位 m。
//   · y 下限钳到 1.5 m（与参考 web-ocean 一致，不做水下视角）；
//   · x/z 按世界周期回绕（地形是可平铺周期瓦片，回绕瞬间画面连续）。
// render_sky 每帧在 linglong_ocean_prepare 之前调用一次；天空视图 LUT 只在
// 高度变化超过相对阈值时重建（见 celestial_ocean.c 的 linglong_ocean_prepare）。
// ===============================================================================
void   linglong_ocean_set_observer(double x, double y, double z);
void   linglong_ocean_get_observer(double *x, double *y, double *z);

// 地形高度（m，相对平均海平面；未启用/未就绪时返回 -WO_ISL_DEEP）。
// 供上层做“相机不钻地”的高度钳制与漫游记账。
double linglong_ocean_terrain_height(double x, double z);

// ===============================================================================
// 随机岛屿地形（参考 linglong.html 的 === [Islands] ===）
//
// 数据：一张可平铺的周期高程图（res²，纹素 world/res m），陆地为正、海底为负；
//       两级“格内最大高程”粗网格（64²/8²）用于射线求交的整格跳过。
// 生成：分帧渐进（每帧预算见 celestial_ocean.c 的 WO_ISL_GEN_BUDGET_MS），
//       生成期间海面照常渲染、地形逐步长出，不会冻结界面。
// 渲染：地形层按 1/div 分辨率求交 + 着色，合成时对轮廓块做全分辨率细化；
//       地形可出现在地平线以上（山脊遮挡日月星），并同时供海面着色取海底高程
//       （浅水色 + 拍岸泡沫）。
// ===============================================================================
typedef struct {
    int32_t  enable;    // 0=纯深水（与移植初版逐位一致）；1=随机岛屿
    int32_t  quality;   // 0=省电(1/3 分辨率 + 近距离远景 LOD) 1=标准(1/2) 2=精细
    int32_t  res;       // 高程图分辨率：512（默认，纹素 50m）/ 1024（纹素 25m）
    uint32_t seed;      // 0 → 内置默认种子；换岛 = seed+1
} Linglong_Terrain_Config;

void    linglong_ocean_terrain_default(Linglong_Terrain_Config *cfg);
void    linglong_ocean_terrain_configure(const Linglong_Terrain_Config *cfg);  // 变更检测 + 懒重建
int32_t linglong_ocean_terrain_progress(void);   // 0..100；100 = 就绪
int32_t linglong_ocean_terrain_ready(void);
void    linglong_ocean_terrain_regenerate(uint32_t seed);   // 立即作废并重新生成

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

// 实时海洋地景：射线步进求交 + 物理着色。
//   · 海面：只写地平线以下的像素；
//   · 岛屿地形：可出现在地平线以上（在天空/天体之后绘制，故正确遮挡日月星）；
//     地形先于海面求交判定，命中则整像素归陆地（陆地在海面之前，与参考一致）。
// 调用方负责在此之前完成天空/恒星等绘制。
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
