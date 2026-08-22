#ifndef __NANO_UI_CLOUD_H__
#define __NANO_UI_CLOUD_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 体积云与天空仿真（port of flower 引擎体积云/大气渲染核心）
//
// 参照 flower（/mnt/d/Desktop/GitRepos/flower，Vulkan 引擎）的
//   - cloud_noise_common.glsl / cloud_basic_noise.glsl / cloud_detailed_noise.glsl
//   - cloud_render_common.glsl / cloud_render_raymarching.glsl
//   - sky_render.glsl / common_shader.glsl / aces.glsl
// 在 CPU（linux tty）上逐像素 ray-march 实现。算法核心与 flower 一致：
//   Perlin-Worley + Worley FBM 三层高度分层云形、双叶相位 + 多重散射近似、
//   HG/Rayleigh 相位大气单次散射 + 多重散射 LUT 的天空模型。
//
// 交互（逻辑键，见 hal_key.h；括注为 tty 物理键位）：
//   左/右          旋转视角（偏航）
//   上/下          俯仰视角
//   7              投影算法循环：透视投影 ↔ 等距鱼眼投影（各自保留 FOV 档位）
//   9              视场角 FOV 循环（当前投影的档位表：透视 12/20/28/38/50°；
//                   鱼眼 60/80/100/130/165°）
//   3（物理键）   手动太阳预设切换（正午→午后→傍晚→日落→月夜；切换即暂停自动）
//   1              云层种类循环：所有云 → 仅低层 → 仅中层 → 低+中 → 无云 → 所有云
//   回车(ENTER)    太阳自动运动 开/暂停（东 -10° → 天顶 → 西 -10°，每帧 +1° 高度角）
//   9              手动太阳预设切换（正午→午后→傍晚→日落→月夜；切换即暂停自动）
//   4 / 6          云量档位 ±（晴空→疏云→半云→多云→阴天→满云，共 6 档）
//   2 / 8          云亮度 ±（0.5~2.0；联动云反照率/太阳辐照/多重散射/环境反射
//                   等介质光学参数，物理地改变透射/反射综合观感）
//   5              云量/亮度复位（并暂停自动、回到正午预设）
//   * (ESC)        返回主菜单
// ===============================================================================

// ===============================================================================
// 体积云/大气渲染核心（与独立 ui_cloud 应用共享同一 ray-march 内核）
//
// 供“玲珑天象仪”（celestial.c）作为新的天空模型调用：
//   - 相机姿态（yaw/pitch/roll，弧度）、投影与视场角由调用方显式给出；
//   - 光源方向/颜色/强度由调用方（天象仪的 where_is_the_sun 提供太阳角度）给出；
//   - 云量 / 云层种类 / 云亮度沿用独立应用中已有的控制参数。
// 平台无关（纯 C + 数学库 + graphics.h 绘制接口）。
// ===============================================================================

// 云层种类掩码（与独立应用“1 键”循环的档位一致）
#define UI_CLOUD_LAYER_LOW  (1)
#define UI_CLOUD_LAYER_MID  (2)
#define UI_CLOUD_LAYER_HIGH (4)
#define UI_CLOUD_LAYER_ALL  (7)

typedef struct UiCloud_Render_Params {
    // ---- 相机（ui_cloud 内部为 y-up 世界，X 东 / Y 上 / Z 南） ----
    float yaw_rad;          // 偏航（弧度）；yaw=0 朝南，+方向逆时针旋转
    float pitch_rad;        // 俯仰（弧度），+上 -下
    float roll_rad;         // 滚转（弧度，绕视线轴；正值为顺时针，与天象仪 fisheye_project 约定一致）
    int   proj;             // 投影：0=透视  1=等距鱼眼（ui_cloud 内部约定）
    int   fov_deg;          // 当前投影的半视场角（度）
    // ---- 时间（驱动云飘移动画，秒） ----
    float app_time_sec;
    // ---- 光源（指向太阳的单位向量，y-up 世界坐标）+ 颜色/强度 ----
    float sun_dx, sun_dy, sun_dz;
    float sun_r, sun_g, sun_b;
    float sun_intensity;
    // ---- 云参数 ----
    float coverage;         // 云量（建议使用 ui_cloud_coverage_for_level 的标定档位值）
    int   layer_mask;       // 云层种类掩码（UI_CLOUD_LAYER_* 组合）
    float brightness;       // 云亮度 0.5~2.0
    int   enable_sun_lens;  // 是否绘制云内核自带的太阳镜头光晕（天象仪集成时关闭，因为它自己画太阳）
} UiCloud_Render_Params;

// 云量档位数（与独立应用 4/6 键的档位表一致，默认 6 档）
int   ui_cloud_coverage_level_num(void);
// 档位 → 云覆盖度标定值（档位钳制到有效范围）
float ui_cloud_coverage_for_level(int level);

// 由太阳高度角（度）推导太阳色温/辐照（线性 RGB + 强度），
// 供天象仪以 where_is_the_sun 提供的太阳角度驱动体积云/大气渲染的光源。
void ui_cloud_sun_color(float elev_deg, float *r, float *g, float *b, float *intensity);

// 将 UiCloud_Render_Params 应用到内部场景并渲染一帧（结果在 ui_cloud 内部缓冲，
// 需调用 ui_cloud_flush 输出到显存；若调用方还要在其上叠加绘制，先 flush 再画）
void ui_cloud_render_core(Nano_GFX *gfx, const UiCloud_Render_Params *p);
// 将最近一帧的渲染结果拷贝到 gfx 的帧缓冲（RGB888 整屏 / 其它色彩模式逐像素回退）
void ui_cloud_flush(Nano_GFX *gfx);

int32_t ui_cloud_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_cloud_event_handler(Key_Event *key_event, Global_State *global_state);
int32_t ui_cloud_render_frame(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
