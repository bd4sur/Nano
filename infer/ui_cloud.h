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
//
// 性能：内部按屏幕 1/2 分辨率 ray-march（160x120 @320x240）后双线性放大，
// 且把 marching 区间裁到当前云层掩码对应的高度带（见 ui_cloud.c 的 CLOUD_OPT_* 档位）；
// 仅低层云 + 半分辨率 + 0.25km 步距在 WASM 单线程下≈12fps（原全层全分辨率≈0.9fps）。
// ===============================================================================

// 云层种类掩码（四层，与参考实现 linglong.html 的按位编号一致：
// bit0 浓积云 / bit1 高积云 / bit2 层积云 / bit3 卷层云）
//   CU 浓积云 云底 ~1.0 km、塔顶 ~4.1 km（平底花椰菜塔，移植自 linglong.html）
//   AC 高积云 3.0–6.2 km（原“低层”）
//   SC 层积云 6.2–9.4 km（原“中层”）
//   CS 卷层云 9.4–11.0 km（原“高层”）
#define UI_CLOUD_LAYER_CU   (1)
#define UI_CLOUD_LAYER_AC   (2)
#define UI_CLOUD_LAYER_SC   (4)
#define UI_CLOUD_LAYER_CS   (8)
#define UI_CLOUD_LAYER_ALL  (15)

// 渲染遍：把“天空/大气”和“体积云”拆成两遍，云层才能在任意时机合成到帧缓冲上
// （照搬 linglong.html 的解耦云层：sky_only + render_decoupled_volumetric_clouds）。
//   0 = BOTH   天空+云一起算（历史行为，独立“体积云”演示 App 用）
//   1 = SKY    只算大气天空（不透明，走 ui_cloud_flush）
//   2 = CLOUDS 只算体积云，输出“预乘覆盖率的显示空间云色 + 云透射率 alpha”，
//              由 ui_cloud_flush_over 叠加到当前帧缓冲（地景/天体之上或之下）
#define UI_CLOUD_PASS_BOTH   (0)
#define UI_CLOUD_PASS_SKY    (1)
#define UI_CLOUD_PASS_CLOUDS (2)

typedef struct UiCloud_Render_Params {
    // ---- 相机（ui_cloud 内部为 y-up 世界，X 东 / Y 上 / Z 南） ----
    float yaw_rad;          // 偏航（弧度）；yaw=0 朝南，+方向逆时针旋转
    float pitch_rad;        // 俯仰（弧度），+上 -下
    float roll_rad;         // 滚转（弧度，绕视线轴；正值为顺时针，与天象仪 fisheye_project 约定一致）
    int   proj;             // 投影：0=透视  1=等距鱼眼（ui_cloud 内部约定）
    int   fov_deg;          // 当前投影的半视场角（度）
    float cam_alt_km;       // 观察高度（km，地面以上；<=0 用内部默认 2.0 km）
    int   pass;             // UI_CLOUD_PASS_*（0 = BOTH，见上）
    // ---- 时间（驱动云飘移动画，秒） ----
    float app_time_sec;
    // ---- 光源（指向太阳的单位向量，y-up 世界坐标）+ 颜色/强度 ----
    float sun_dx, sun_dy, sun_dz;
    float sun_r, sun_g, sun_b;
    float sun_intensity;
    // ---- 云参数 ----
    float coverage;         // 云量（建议使用 ui_cloud_coverage_for_level 的标定档位值）
    int   layer_mask;       // 云层种类掩码（UI_CLOUD_LAYER_* 组合）
    float brightness;       // 云亮度 0.5~2.0（天象仪传入 ui_cloud_brightness_for_sun_alt() 的
                            // 自动调制值；独立应用仍可用 2/8 键手动调节）
    int   enable_sun_lens;  // 是否求太阳方向的云透射率并绘制云内核自带的太阳镜头光晕
                            // （天象仪：置 1 只为了拿可见度，光晕由 render_sky 末尾统一叠加；
                            //   独立体积云 App：置 1 则 PASS_BOTH 内叠 HDR 光晕）
    // ---- 性能档位覆盖（0 = 用 ui_cloud 内部默认值，见 CLOUD_OPT_* 宏） ----
    int   render_scale;     // 内部渲染降采样倍数（1=全分辨率，2=半分辨率…）
    float step_nominal_km;  // 云区裁剪后的目标采样步距（km）
} UiCloud_Render_Params;

// 云量档位数（与独立应用 4/6 键的档位表一致，默认 6 档）
int   ui_cloud_coverage_level_num(void);
// 档位名称（“晴空/疏云/半云/多云/阴天/满云”；越界钳制到有效范围）
const wchar_t *ui_cloud_coverage_level_name(int level);
// 档位 → 云覆盖度标定值（档位钳制到有效范围）
float ui_cloud_coverage_for_level(int level);

// 由太阳高度角（度）推导太阳色温/辐照（线性 RGB + 强度），
// 供天象仪以 where_is_the_sun 提供的太阳角度驱动体积云/大气渲染的光源。
void ui_cloud_sun_color(float elev_deg, float *r, float *g, float *b, float *intensity);

// 云亮度（0.5~2.0）随太阳高度的经验调制（与 bd4sur.github.io/html/linglong.html 同式）：
//   brightness = 0.75·tanh(0.2·h − 0.6) + 1.25    （h = 太阳高度角，度）
// 值域恰好落在亮度区间 [0.5, 2.0]：h≈3° → 1.25；h≳25° 饱和到上限 2.0（云白亮）；
// 日落后急降；天文暮光（h≲−18°）之后压到下限 0.5。
// 与 ui_cloud_sun_color 一样是“每帧按太阳位置重算”的曲线：亮度经 cloud_apply_brightness
// 联动反照率/直射辐照/多重散射/环境补光等介质光学参数，而不是事后乘颜色。
float ui_cloud_brightness_for_sun_alt(float elev_deg);

// 将 UiCloud_Render_Params 应用到内部场景并渲染一帧（结果在 ui_cloud 内部缓冲，
// 需调用 ui_cloud_flush / ui_cloud_flush_over 输出到显存；若调用方还要在其上叠加
// 绘制，先 flush 再画）
void ui_cloud_render_core(Nano_GFX *gfx, const UiCloud_Render_Params *p);
// 将最近一帧的渲染结果拷贝到 gfx 的帧缓冲（RGB888 整屏 / 其它色彩模式逐像素回退）
// —— 用于 PASS_SKY / PASS_BOTH（不透明）。
void ui_cloud_flush(Nano_GFX *gfx);
// 将最近一帧 PASS_CLOUDS 的结果按标准 alpha 合成叠加到 gfx 当前帧缓冲：
//   dst = dst·T + premul_rgb     （T = 云透射率，premul_rgb = 云色·(1−T)）
// 天空、天体、地景都可以先在帧缓冲里，再调用本函数让云遮住它们（云上俯视），
// 也可以先调用本函数再画地景（地面视角，近景遮远云）。
void ui_cloud_flush_over(Nano_GFX *gfx);
// 当前启用云层里最低的云底（km，地面以上）；无云（掩码 0）或尚未初始化时返回 -1。
// 用于决定“云比地景近还是远”：观察高度 >= 该值时云在脚下，必须先画地景再合成云。
float ui_cloud_lowest_layer_base_km(void);

// ===============================================================================
// 太阳镜头光晕（屏幕空间叠加光；参照 linglong.html 的 cloud_sky_lens_hdr）
//
// 天象仪在天空/云/地景全部画完之后调用本函数，把 6 个「离轴色散光斑」以加法叠到
// 当前帧缓冲（不遮挡、不调色）：镜头光晕是光在镜筒内多次反射的产物，属于屏幕空间
// 叠加光。太阳核心盘/星芒仍由天象仪自己绘制（参考实现的 lens 也已移出星芒）；
// 参考实现中默认关闭的太阳周围散射光晕（CLOUD_SKY_SUN_GLOW=0）同样不绘制。
//   sun_ndc_x/sun_ndc_y：太阳的屏幕 NDC（±1，x 右正、y 上正），由 fisheye_project 得出；
//   sun_vis：沿太阳方向的可见度（体积云透射率；无云/其它散射模型传 1.0）。
// 成本：只遍历各光斑包围盒（合计约 2k 像素，320x240）。
// ===============================================================================
void  ui_cloud_lens_flare_draw(Nano_GFX *gfx, float sun_ndc_x, float sun_ndc_y, float sun_vis);
// 沿太阳方向的云透射率（0..1）：体积云天空（模型 4）每帧算好；无云或未渲染时为 0。
float ui_cloud_sun_visibility(void);

int32_t ui_cloud_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_cloud_event_handler(Key_Event *key_event, Global_State *global_state);
int32_t ui_cloud_render_frame(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
