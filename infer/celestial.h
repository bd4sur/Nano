#ifndef __NANO_CELESTIAL_H__
#define __NANO_CELESTIAL_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "graphics.h"

// 若干视觉参数
#define LINGLONG_STAR_BURST_RADIUS        (60)
#define LINGLONG_STAR_BURST_DECAY         (0.94f)
#define LINGLONG_HORIZON_BLUR_MARGIN      (10.0f)
#define LINGLONG_CELESTIAL_CIRCLE_POINTS  (180)
#define LINGLONG_HORIZONTAL_CIRCLE_POINTS (180)
#define LINGLONG_ECLIPTIC_CIRCLE_POINTS   (360)
#define LINGLONG_SUN_RADIUS_MIN           (3.0f)
#define LINGLONG_SUN_GLOW_RADIUS          (4)
#define LINGLONG_MOON_RADIUS_MIN          (16.0f)
#define LINGLONG_STAR_GLOW_RADIUS         (0)
// 是否开启动态地景：动态地景需动态分配大量堆内存，若内存紧张，可关闭。同时关闭透视转鱼眼功能，详见代码。
// M5Core2/CoreS3（ESP32）：固件分区 6.5MB 约束 + 堆内存紧张，关闭（与 linglong_m5core2
// 仓库验证过的固件行为一致；该平台的 128x128 鱼眼纹理头亦不含 FLAT 纹理）
#if NANO_IS_M5ESP
    #define LINGLONG_ENABLE_DYNAMIC_LANDSCAPE (0)
#else
    #define LINGLONG_ENABLE_DYNAMIC_LANDSCAPE (1)
#endif

// 二次散射模型相关参数
#define LINGLONG_SCATTER_MARCHRAY_NV  (36)
#define LINGLONG_SCATTER_MARCHRAY_NL  (1)
#define LINGLONG_SCATTER_MARCHRAY_V2  (3)
#define LINGLONG_SCATTER_MARCHRAY_NV2 (2)
#define LINGLONG_SCATTER_MARCHRAY_NL2 (2)

typedef struct Linglong_Config {
    int32_t fb_width;
    int32_t fb_height;
    float sky_radius;
    float center_x;
    float center_y;
    float view_alt;
    float view_azi;
    float view_roll;
    float view_f;

    int32_t year;
    int32_t month;
    int32_t day;
    int32_t hour;
    int32_t minute;
    int32_t second;
    double timezone;
    double longitude;
    double latitude;

    int32_t downsampling_factor;     // 降采样因子（设为0为自动，建议设为2）
    int32_t enable_opt_sym;          // 【废弃】是否启用基于对称性的渲染优化（以画质为代价）
    int32_t enable_opt_lut;          // 是否启用查找表计算加速（以画质为代价）
    int32_t enable_opt_bilinear;     // 是否启用双线性插值以优化画质
 
    int32_t projection;              // 投影算法（0-鱼眼；1-线性透视）
    int32_t sky_model;               // 选择天空模型（0-不启用散射；1-简单散射；2-一次散射；3-二次散射；4-体积云+大气渲染；5-海洋天空）
    int32_t landscape_index;         // 选择地景（0-不启用，地景设为纯黑；1-鱼眼贴图；2-动态地景；3-实时海洋）
    int32_t enable_equatorial_coord; // 是否启用赤道坐标圈
    int32_t enable_horizontal_coord; // 是否启用地平坐标圈（0-不启用；1-仅方位角文字；2-方位角+坐标圈）
    int32_t enable_star_burst;       // 是否启用星芒效果
    int32_t enable_star_name;        // 是否显示恒星名称（0-不显示；1-除行星；2-仅行星；3-全部）
    int32_t enable_planet;           // 是否显示大行星
    int32_t enable_ecliptic_circle;  // 是否显示黄道
    int32_t enable_att_indicator;    // 是否显示姿态指示标记
    int32_t enable_tracking_sun;     // 视线偏航角是否跟踪太阳

    // 体积云参数（仅 sky_model==4 时有效；对应 ui_cloud.c 中已有的云量/云层/精度控制）。
    // 云亮度不在此列：它每帧由太阳高度自动调制（ui_cloud_brightness_for_sun_alt）。
    int32_t cloud_coverage_level;    // 云量档位 0-5（晴空→疏云→半云→多云→阴天→满云）
    int32_t cloud_layer_mask;        // 云层种类掩码；bit0-低层 bit1-中层 bit2-高层（7=全部，0=无云）
    int32_t cloud_precision_level;   // 云精度档位 0-2（高/中/低：内部渲染降采样倍数 + ray-march 步距）

    // 海洋/观察高度参数（观察高度对 sky_model==4/5 或 landscape_index==3 均有效；
    // 档位表见 celestial_ocean.h）。体积云用它决定相机海拔（天空+云层 ray-march 同一来源）。
    // 其余海况（风/泡沫/浑浊度/霾/能见度）按需求固定为模块默认值，不开放给用户。
    int32_t ocean_height_level;      // 观察高度档位 0-7（5m/50m/200m/1000m/4000m/8000m/12000m/20000m；默认 0=5m）
    int32_t ocean_swell_hs_level;    // 涌浪波高档位 0-5（无/0.5/1.0/1.8/2.5/3.5 m；默认 3=1.8m）
    int32_t ocean_swell_len_level;   // 涌浪波长档位 0-6（30/60/90/120/160/220/300 m；默认 3=120m）

    // 以下与天空渲染无关（非render_sky参数）
    int32_t enable_imu;              // 是否启用IMU（使视角随机器姿态旋转）

    float lut_alt_step;              // Inscatter LUT 高度角步长（度），默认 0.5
    float lut_azi_step;              // Inscatter LUT 方位角步长（度），默认 0.5
    float lut_transmittance_step;    // Transmittance LUT 天顶角步长（度），默认 0.5
} Linglong_Config;



void linglong_init(Linglong_Config *cfg);
void linglong_free(void);

// ===============================================================================
// 「云类」（云层掩码）与「云精度」档位表
// 玲珑仪触屏控制面板的对应按钮与控制逻辑共用这一份唯一真源（选项值 + 显示名）。
// ===============================================================================
typedef struct {
    int32_t        value;   // 云类：云层掩码（UI_CLOUD_LAYER_*）；云精度：档位序号
    const wchar_t *name;    // 按钮上显示的名称
} Linglong_Cloud_Option;

// 返回档位表首地址，*count 写入项数（不会为 0）
const Linglong_Cloud_Option *linglong_cloud_layer_options(int32_t *count);
const Linglong_Cloud_Option *linglong_cloud_precision_options(int32_t *count);

// 鱼眼/透视投影互为逆运算（屏幕坐标 ↔ 地平天球笛卡尔坐标）；celestial_ocean.c 复用
void fisheye_unproject(
    float scr_x, float scr_y,
    float radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f, int32_t projection,
    float *x, float *y, float *z
);

// 屏幕方向图（celestial.c 内缓存：每个屏幕像素 → 地平系方向）：视角不变时完全可复用。
// refresh 按需重建（参数未变时立即返回），ready 校验尺寸与有效性（宽高不符返回 0）。
void    fisheye_dir_map_refresh(
    uint32_t width, uint32_t height,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f, int32_t projection
);
int32_t fisheye_dir_map_ready(int32_t width, int32_t height);
void    fisheye_dir_map_get(int32_t x, int32_t y, float *rx, float *ry, float *rz);

void render_sky(Nano_GFX *gfx,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double timezone, double longitude, double latitude,
    int32_t downsampling_factor,     // 降采样因子（设为0为自动，建议设为2）
    int32_t enable_opt_sym,          // 【废弃】是否启用基于对称性的渲染优化（以画质为代价）
    int32_t enable_opt_lut,          // 是否启用查找表计算加速（以画质为代价）
    int32_t enable_opt_bilinear,     // 是否启用双线性插值以优化画质

    int32_t projection,              // 投影算法（0-鱼眼；1-线性透视）
    int32_t sky_model,               // 选择天空模型（0-不启用散射；1-简单散射；2-一次散射；3-二次散射；4-体积云+大气渲染；5-海洋天空）
    int32_t landscape_index,         // 选择地景（0-不启用，地景设为纯黑；1-鱼眼贴图；2-动态地景；3-实时海洋）
    int32_t enable_equatorial_coord, // 是否启用赤道坐标圈
    int32_t enable_horizontal_coord, // 是否启用地平坐标圈（0-不启用；1-仅方位角文字；2-方位角+坐标圈）
    int32_t enable_star_burst,       // 是否启用星芒效果
    int32_t enable_star_name,        // 是否显示恒星名称（0-不显示；1-除行星；2-仅行星；3-全部）
    int32_t enable_planet,           // 是否显示大行星
    int32_t enable_ecliptic_circle,  // 是否显示黄道
    int32_t enable_att_indicator,    // 是否显示姿态指示标记
    int32_t enable_tracking_sun,     // 视线偏航角是否跟踪太阳
    // 体积云参数（仅 sky_model==4 时有效）
    int32_t cloud_coverage_level,    // 云量档位 0-5
    int32_t cloud_layer_mask,        // 云层种类掩码（bit0/1/2）
    int32_t cloud_precision_level,   // 云精度档位 0-2（高/中/低）
    // 海洋/观察高度参数（观察高度对 sky_model==4/5 或 landscape_index==3 均有效）
    int32_t ocean_height_level,      // 观察高度档位 0-7（体积云也用它决定相机海拔）
    int32_t ocean_swell_hs_level,    // 涌浪波高档位 0-5
    int32_t ocean_swell_len_level    // 涌浪波长档位 0-6
);

#ifdef __cplusplus
}
#endif

#endif
